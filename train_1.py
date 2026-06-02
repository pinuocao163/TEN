import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from llm_reasoning import LLMReasoningChain, cognitive_feature_dim
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime

def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset('/data/zzb/BaseLine/SDT/data/meld_multimodal_features.pkl')
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('/data/zzb/BaseLine/SDT/data/meld_multimodal_features.pkl', train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def weighted_kl_loss(log_pred, target, mask, weight=None):
    mask_ = mask.view(-1).bool()
    if mask_.sum().item() == 0:
        return log_pred.sum() * 0.0

    log_pred = log_pred[mask_]
    target = target[mask_].detach()
    target = torch.clamp(target, min=1e-8)
    target = target / target.sum(dim=-1, keepdim=True)
    loss = torch.nn.functional.kl_div(log_pred, target, reduction='none').sum(dim=-1)
    if weight is not None:
        weight = weight.view(-1)[mask_].detach()
        loss = loss * torch.clamp(weight, min=0.0)
        return loss.sum() / torch.clamp(weight.sum(), min=1.0)
    return loss.mean()


def cognitive_gate_target(llm_features, n_classes):
    appraisal = llm_features[:, :, n_classes:n_classes + 4]
    modality_hint = llm_features[:, :, n_classes + 4:n_classes + 7]
    confidence = llm_features[:, :, n_classes + 7:n_classes + 8]
    if llm_features.size(-1) > n_classes + 8:
        rag_quality = llm_features[:, :, n_classes + 8:n_classes + 9]
    else:
        rag_quality = confidence * 0.75
    if llm_features.size(-1) > n_classes + 9:
        vad_guidance = llm_features[:, :, n_classes + 9:n_classes + 12]
    else:
        vad_guidance = torch.zeros_like(modality_hint)

    valence = appraisal[:, :, 0:1]
    arousal = appraisal[:, :, 1:2]
    dominance = appraisal[:, :, 2:3]
    certainty = appraisal[:, :, 3:4]
    pos_activation = vad_guidance[:, :, 0:1].abs()
    neg_control = vad_guidance[:, :, 1:2].abs()
    neg_low = vad_guidance[:, :, 2:3].abs()

    vad_modality = torch.cat([
        certainty.abs() + neg_low + (1.0 - arousal.abs()).clamp(min=0.0) * 0.25,
        arousal.abs() + neg_control,
        valence.clamp(min=0.0) * 0.25 + pos_activation + dominance.abs() * 0.10,
    ], dim=-1)
    vad_modality = torch.clamp(vad_modality, min=1e-6)
    vad_modality = vad_modality / vad_modality.sum(dim=-1, keepdim=True)

    modality = 0.65 * modality_hint + 0.35 * vad_modality
    modality = torch.clamp(modality, min=1e-6)
    modality = modality / modality.sum(dim=-1, keepdim=True)

    vad_strength = torch.clamp(appraisal[:, :, :3].abs().mean(dim=-1, keepdim=True) +
                               vad_guidance.abs().mean(dim=-1, keepdim=True), 0.0, 1.0)
    reason_prior = confidence * (0.5 + 0.5 * rag_quality) * (0.12 + 0.18 * vad_strength)
    reason_prior = torch.clamp(reason_prior, min=0.0, max=0.30)
    target = torch.cat([modality * (1.0 - reason_prior), reason_prior], dim=-1)
    return target / torch.clamp(target.sum(dim=-1, keepdim=True), min=1e-6)


def vad_aware_contrastive_loss(features, labels, mask, llm_features, llm_mask, n_classes, temperature=0.2, margin=0.2):
    valid = mask.view(-1).bool()
    if llm_features is not None and llm_mask is not None:
        valid = valid & llm_mask.view(-1).bool()
    reps = features.reshape(-1, features.size(-1))[valid]
    labs = labels.view(-1)[valid]
    if reps.size(0) <= 1:
        return features.sum() * 0.0

    reps = torch.nn.functional.normalize(reps, dim=-1)
    sim = torch.matmul(reps, reps.t()) / temperature
    logits_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    same_label = labs.unsqueeze(0).eq(labs.unsqueeze(1)) & logits_mask

    if llm_features is not None:
        vad = llm_features[:, :, n_classes:n_classes + 3].reshape(-1, 3)[valid]
        vad_dist = torch.cdist(vad, vad, p=2)
        vad_sim = torch.exp(-vad_dist)
    else:
        vad_sim = torch.ones_like(sim)

    pos_weight = same_label.float() * (0.5 + 0.5 * vad_sim)
    pos_count = pos_weight.sum(dim=1)
    log_prob = sim - torch.logsumexp(sim.masked_fill(~logits_mask, -1e4), dim=1, keepdim=True)
    pos_loss = -(pos_weight * log_prob).sum(dim=1) / torch.clamp(pos_count, min=1.0)
    pos_loss = pos_loss[pos_count > 0].mean() if (pos_count > 0).any() else sim.sum() * 0.0

    confusion_pairs = [(0, 4), (3, 5), (2, 1)] if n_classes == 6 else []
    neg_mask = torch.zeros_like(same_label)
    for a, b in confusion_pairs:
        if a < n_classes and b < n_classes:
            pair = ((labs.unsqueeze(0).eq(a) & labs.unsqueeze(1).eq(b)) |
                    (labs.unsqueeze(0).eq(b) & labs.unsqueeze(1).eq(a)))
            neg_mask = neg_mask | pair
    neg_mask = neg_mask & logits_mask
    if neg_mask.any():
        raw_sim = torch.matmul(reps, reps.t())
        neg_weight = 0.5 + 0.5 * (1.0 - vad_sim)
        neg_loss = (torch.relu(raw_sim - margin) * neg_weight * neg_mask.float()).sum() / \
                   torch.clamp(neg_mask.float().sum(), min=1.0)
    else:
        neg_loss = sim.sum() * 0.0
    return pos_loss + neg_loss


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False,
                        gamma_1=1.0, gamma_2=1.0, gamma_3=1.0, llm_chain=None,
                        llm_loss_weight=0.0, use_llm_reasoning=False,
                        llm_reliability_weight=0.0, vad_contrast_weight=0.0,
                        llm_min_quality=0.0, llm_min_confidence=0.0,
                        grad_clip=0.0):
    losses, preds, labels, masks = [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        dialogue_ids = data[-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        llm_features, llm_mask, llm_fusion_mask = None, None, None
        llm_quality, llm_confidence = None, None
        if llm_chain is not None and use_llm_reasoning:
            llm_features, llm_mask = llm_chain.batch_cognitive_features(
                dialogue_ids, lengths, umask.size(1), textf.device)
            if llm_features is not None:
                n_classes = model.n_classes
                llm_confidence = llm_features[:, :, n_classes + 7]
                if llm_features.size(-1) > n_classes + 8:
                    llm_quality = llm_features[:, :, n_classes + 8]
                else:
                    llm_quality = llm_confidence * 0.75
                high_quality = llm_quality.ge(llm_min_quality) & llm_confidence.ge(llm_min_confidence)
                llm_fusion_mask = llm_mask * high_quality.float()

        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = model(
            textf, visuf, acouf, umask, qmask, lengths,
            llm_features=llm_features if use_llm_reasoning else None,
            llm_mask=llm_fusion_mask if use_llm_reasoning else None)
        
        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])
        
        loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                gamma_2 * (loss_function(lp_1, labels_, umask) + loss_function(lp_2, labels_, umask) + loss_function(lp_3, labels_, umask)) + \
               gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) + kl_loss(kl_lp_2, kl_p_all, umask) + kl_loss(kl_lp_3, kl_p_all, umask))

        llm_prob = None
        if llm_features is not None:
            llm_prob = llm_features[:, :, :all_prob.size(2)] if use_llm_reasoning else llm_features

        if train and llm_loss_weight > 0.0 and llm_prob is not None:
            if use_llm_reasoning and llm_quality is not None and llm_confidence is not None:
                high_quality = llm_quality.ge(llm_min_quality) & llm_confidence.ge(llm_min_confidence)
                distill_weight = llm_mask * high_quality.float() * llm_quality * llm_confidence
            else:
                distill_weight = llm_mask
            llm_loss = weighted_kl_loss(lp_all, llm_prob.reshape(-1, llm_prob.size()[2]), llm_mask, distill_weight)
            loss = loss + llm_loss_weight * llm_loss

        if train and use_llm_reasoning and llm_reliability_weight > 0.0 and \
                llm_features is not None and model.last_reliability_weights is not None:
            hint = cognitive_gate_target(llm_features, all_prob.size(2))
            rel_loss = ((model.last_reliability_weights - hint) ** 2).sum(dim=-1)
            if llm_quality is not None and llm_confidence is not None:
                rel_weight = llm_fusion_mask * torch.clamp(llm_quality * llm_confidence, min=0.0, max=1.0)
            else:
                rel_weight = llm_mask
            rel_loss = (rel_loss * rel_weight).sum() / torch.clamp(rel_weight.sum(), min=1.0)
            loss = loss + llm_reliability_weight * rel_loss

        if train and vad_contrast_weight > 0.0 and model.last_contrast_features is not None:
            contrast_mask = llm_fusion_mask if llm_fusion_mask is not None else llm_mask
            if llm_quality is not None and llm_confidence is not None:
                contrast_mask = llm_mask * (llm_quality.ge(llm_min_quality) & llm_confidence.ge(llm_min_confidence)).float()
            contrast_loss = vad_aware_contrastive_loss(
                model.last_contrast_features, label, umask, llm_features, contrast_mask, all_prob.size(2))
            loss = loss + vad_contrast_weight * contrast_loss

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        if train:
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            if grad_clip > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    continue
            optimizer.step()
        losses.append(loss.item()*masks[-1].sum())

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)  
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=150, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--llm-cache', default='', help='offline RAG-LLM cognitive reasoning cache: json/jsonl/csv/pkl')
    parser.add_argument('--llm-loss-weight', type=float, default=0.0, help='quality-weighted KL loss for RAG-LLM soft-label guidance')
    parser.add_argument('--llm-label-confidence', type=float, default=0.85, help='confidence used when the LLM cache only stores hard labels')
    parser.add_argument('--llm-temperature', type=float, default=1.0, help='temperature applied to LLM probability distributions')
    parser.add_argument('--require-llm-cache', action='store_true', default=False, help='raise an error when --llm-cache is missing')
    parser.add_argument('--use-llm-reasoning', action='store_true', default=False, help='enable integrated RAG/VAD cognitive residual fusion')
    parser.add_argument('--llm-reliability-weight', type=float, default=0.02, help='weight for aligning fusion reliability with VAD and RAG-guided modality hints')
    parser.add_argument('--vad-contrast-weight', type=float, default=0.0, help='weight for VAD-aware supervised contrastive loss on cognitive fusion states')
    parser.add_argument('--llm-min-quality', type=float, default=0.55, help='minimum reasoning quality for LLM distillation and VAD contrastive supervision')
    parser.add_argument('--llm-min-confidence', type=float, default=0.80, help='minimum LLM confidence for LLM distillation and VAD contrastive supervision')
    parser.add_argument('--llm-residual-init', type=float, default=0.05, help='initial strength of RAG-LLM cognitive residual fusion')
    parser.add_argument('--grad-clip', type=float, default=0.0, help='gradient clipping norm; 0 disables clipping')

    args = parser.parse_args()
    if args.llm_cache and not args.use_llm_reasoning:
        args.use_llm_reasoning = True
        print('Enable integrated RAG/VAD reasoning because --llm-cache is provided.')
    today = datetime.datetime.now()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    print('temp {}'.format(args.temp))

    llm_feature_dim = cognitive_feature_dim(n_classes) if args.use_llm_reasoning else 0
    model = Transformer_Based_Model(args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
                                        n_classes=n_classes,
                                        hidden_dim=args.hidden_dim,
                                        n_speakers=n_speakers,
                                        dropout=args.dropout,
                                        llm_feature_dim=llm_feature_dim,
                                        use_llm_reasoning=args.use_llm_reasoning,
                                        llm_residual_init=args.llm_residual_init)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    if cuda:
        model.cuda()
        
    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=0)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        print("There is no such dataset")

    llm_chain = None
    if args.llm_cache:
        if not os.path.exists(args.llm_cache):
            message = (
                'LLM reasoning cache not found: {}. Continue without LLM reasoning. '
                'Generate it from data/{}_llm_prompts.jsonl, or add --require-llm-cache to fail fast.'
            ).format(args.llm_cache, args.Dataset.lower())
            if args.require_llm_cache:
                raise FileNotFoundError(message)
            print(message)
            args.llm_loss_weight = 0.0
        else:
            llm_chain = LLMReasoningChain(args.llm_cache, args.Dataset, n_classes,
                                          label_confidence=args.llm_label_confidence,
                                          temperature=args.llm_temperature)
            train_hit, train_total = llm_chain.coverage(train_loader)
            test_hit, test_total = llm_chain.coverage(test_loader)
            print('LLM reasoning cache coverage: train {}/{}, test {}/{}'.format(
                train_hit, train_total, test_hit, test_total))

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    if n_epochs <= 0:
        print('No training epochs requested. Exit after setup checks.')
        raise SystemExit(0)

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(
            model, loss_function, kl_loss, train_loader, e, optimizer, True,
            llm_chain=llm_chain, llm_loss_weight=args.llm_loss_weight,
            use_llm_reasoning=args.use_llm_reasoning,
            llm_reliability_weight=args.llm_reliability_weight,
            vad_contrast_weight=args.vad_contrast_weight,
            llm_min_quality=args.llm_min_quality,
            llm_min_confidence=args.llm_min_confidence,
            grad_clip=args.grad_clip)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(
            model, loss_function, kl_loss, valid_loader, e,
            llm_chain=llm_chain, use_llm_reasoning=args.use_llm_reasoning,
            llm_min_quality=args.llm_min_quality,
            llm_min_confidence=args.llm_min_confidence)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(
            model, loss_function, kl_loss, test_loader, e,
            llm_chain=llm_chain, use_llm_reasoning=args.use_llm_reasoning,
            llm_min_quality=args.llm_min_quality,
            llm_min_confidence=args.llm_min_confidence)
        all_fscore.append(test_fscore)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if args.tensorboard:
            writer.add_scalar('test: accuracy', test_acc, e)
            writer.add_scalar('test: fscore', test_fscore, e)
            writer.add_scalar('train: accuracy', train_acc, e)
            writer.add_scalar('train: fscore', train_fscore, e)

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1)%10 == 0:
            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))


    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))
    
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    os.makedirs(result_dir, exist_ok=True)
    record_path = os.path.join(result_dir, "record_{}_{}_{}.pk".format(today.year, today.month, today.day))

    if not os.path.exists(record_path):
        with open(record_path, 'wb') as f:
            pk.dump({}, f)
    with open(record_path, 'rb') as f:
        record = pk.load(f)
    key_ = 'name_'
    if record.get(key_, False):
        record[key_].append(max(all_fscore))
    else:
        record[key_] = [max(all_fscore)]
    if record.get(key_+'record', False):
        record[key_+'record'].append(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    else:
        record[key_+'record'] = [classification_report(best_label, best_pred, sample_weight=best_mask,digits=4)]
    with open(record_path, 'wb') as f:
        pk.dump(record, f)

    print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))

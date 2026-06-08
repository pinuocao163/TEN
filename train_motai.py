import argparse
import datetime
import os
import pickle as pk
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedKLDivLoss, MaskedNLLLoss
from model_motai import ModalityAblationModel, modality_name, normalize_modalities


def get_train_valid_sampler(trainset, valid=0.0):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_meld_loaders(batch_size=32, valid=0.0, num_workers=0, pin_memory=False):
    dataset_path = '/data/zzb/BaseLine/SDT/data/meld_multimodal_features.pkl'
    trainset = MELDDataset(dataset_path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers,
                              pin_memory=pin_memory)
    testset = MELDDataset(dataset_path, train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_iemocap_loaders(batch_size=32, valid=0.0, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers,
                              pin_memory=pin_memory)
    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def active_indices(modalities):
    mods = normalize_modalities(modalities)
    return [idx for idx, key in enumerate(('t', 'a', 'v')) if key in mods]


def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, active_branch_ids,
                        optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0,
                        gamma_3=1.0, grad_clip=0.0, cuda=False):
    losses, preds, labels, masks = [], [], [], []
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        batch = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        textf, visuf, acouf, qmask, umask, label = batch
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        outputs = model(textf, visuf, acouf, umask, qmask, lengths)
        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
            kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob = outputs

        branch_log_probs = [log_prob1, log_prob2, log_prob3]
        branch_kl_log_probs = [kl_log_prob1, kl_log_prob2, kl_log_prob3]
        labels_ = label.view(-1)

        lp_all = all_log_prob.view(-1, all_log_prob.size(2))
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size(2))
        loss = gamma_1 * loss_function(lp_all, labels_, umask)

        if active_branch_ids:
            uni_loss = 0.0
            sd_loss = 0.0
            for branch_idx in active_branch_ids:
                lp_branch = branch_log_probs[branch_idx].view(-1, branch_log_probs[branch_idx].size(2))
                kl_lp_branch = branch_kl_log_probs[branch_idx].view(-1, branch_kl_log_probs[branch_idx].size(2))
                uni_loss = uni_loss + loss_function(lp_branch, labels_, umask)
                sd_loss = sd_loss + kl_loss(kl_lp_branch, kl_p_all, umask)
            loss = loss + gamma_2 * uni_loss + gamma_3 * sd_loss

        pred_ = torch.argmax(all_prob.view(-1, all_prob.size(2)), 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        if train:
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if grad_clip > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad()
                    continue
            optimizer.step()

        losses.append(loss.item() * masks[-1].sum())

    if not preds:
        return float('nan'), float('nan'), [], [], [], float('nan')

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)
    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--temp', type=int, default=1)
    parser.add_argument('--Dataset', default='IEMOCAP', choices=['IEMOCAP', 'MELD'])
    parser.add_argument('--modalities', default='text_audio',
                        help='text/audio/visual/text_audio/text_visual/audio_visual')
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--valid', type=float, default=0.0)
    return parser


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    args.modalities = modality_name(args.modalities)
    active_branch_ids = active_indices(args.modalities)

    today = datetime.datetime.now()
    print(args)
    print('Active modalities: {}'.format(args.modalities))

    cuda = torch.cuda.is_available() and not args.no_cuda
    print('Running on GPU' if cuda else 'Running on CPU')

    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 7 if args.Dataset == 'MELD' else 6

    model = ModalityAblationModel(
        args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
        n_classes=n_classes, hidden_dim=args.hidden_dim, n_speakers=n_speakers,
        dropout=args.dropout, modalities=args.modalities)

    print('total parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    print('training parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if cuda:
        model.cuda()

    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_meld_loaders(
            valid=args.valid, batch_size=args.batch_size, num_workers=0)
    else:
        loss_weights = torch.FloatTensor([
            1 / 0.086747,
            1 / 0.144406,
            1 / 0.227883,
            1 / 0.160585,
            1 / 0.127711,
            1 / 0.252668,
        ])
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_iemocap_loaders(
            valid=args.valid, batch_size=args.batch_size, num_workers=0)

    best_fscore, best_label, best_pred, best_mask = None, None, None, None
    all_fscore = []
    all_acc = []

    if args.epochs <= 0:
        print('No training epochs requested. Exit after setup checks.')
        raise SystemExit(0)

    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore = train_or_eval_model(
            model, loss_function, kl_loss, train_loader, e, active_branch_ids,
            optimizer=optimizer, train=True, grad_clip=args.grad_clip, cuda=cuda)
        valid_loss, valid_acc, _, _, _, valid_fscore = train_or_eval_model(
            model, loss_function, kl_loss, valid_loader, e, active_branch_ids, cuda=cuda)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(
            model, loss_function, kl_loss, test_loader, e, active_branch_ids, cuda=cuda)

        all_fscore.append(test_fscore)
        all_acc.append(test_acc)
        if best_fscore is None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, '
              'valid_loss: {}, valid_acc: {}, valid_fscore: {}, '
              'test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(
                  e + 1, train_loss, train_acc, train_fscore,
                  valid_loss, valid_acc, valid_fscore,
                  test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))

        if (e + 1) % 10 == 0 and best_label is not None:
            print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))

    print('Test performance..')
    print('Best Acc: {}'.format(max(all_acc)))
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    os.makedirs(result_dir, exist_ok=True)
    timestamp = today.strftime('%Y%m%d_%H%M%S')
    record_path = os.path.join(
        result_dir,
        'record_motai_{}_{}_{}_{}.pk'.format(
            args.Dataset.lower(), args.modalities, timestamp, os.getpid()))

    report = classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)
    matrix = confusion_matrix(best_label, best_pred, sample_weight=best_mask)
    record = {
        'dataset': args.Dataset,
        'modalities': args.modalities,
        'best_fscore': max(all_fscore),
        'best_acc': max(all_acc),
        'best_epoch': all_fscore.index(max(all_fscore)) + 1,
        'report': report,
        'confusion_matrix': matrix,
        'args': vars(args),
    }
    with open(record_path, 'wb') as f:
        pk.dump(record, f)
    print('Saved record to {}'.format(record_path))
    print(report)
    print(matrix)

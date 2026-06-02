import argparse
import json
import os
import pickle
import re
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from export_llm_prompts import DATA_PATHS, speaker_name
from llm_reasoning import APPRAISAL_PRIORS, EMOTION_LABELS, MODALITY_PRIORS


LABEL_ALIASES = {
    'sadness': 'sad',
    'joyful': 'joy',
    'anger': 'angry',
    'frustration': 'frustrated',
}


def load_done(output):
    done = set()
    if not os.path.exists(output):
        return done
    with open(output, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add((str(item['vid']), int(item['turn_id'])))
            except Exception:
                continue
    return done


def normalise_label(label, labels):
    if label is None:
        return None
    label = str(label).strip().lower()
    if label in labels:
        return label
    label = LABEL_ALIASES.get(label, label)
    if label in labels:
        return label
    if label == 'joy' and 'happy' in labels:
        return 'happy'
    for candidate in labels:
        if candidate in label:
            return candidate
    return None


def fallback_appraisal(label, confidence):
    prior = APPRAISAL_PRIORS.get(label, [0.0, 0.0, 0.0, confidence])
    return {
        'valence': prior[0],
        'arousal': prior[1],
        'dominance': prior[2],
        'certainty': confidence if prior[3] == 0.0 else prior[3],
    }


def fallback_modality_hint(label):
    prior = MODALITY_PRIORS.get(label, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    total = sum(prior)
    return {
        'text': prior[0] / total,
        'audio': prior[1] / total,
        'visual': prior[2] / total,
    }


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    match = re.search(r'\{.*?\}', text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            return {}
    return {}


def extract_key_value(text, key):
    match = re.search(r'{}\s*[:=]\s*([^\n,;}}]+)'.format(key), text, flags=re.I)
    if match:
        return match.group(1).strip().strip('"\'')
    return None


def format_stats(stats):
    if not isinstance(stats, dict):
        return 'audio_energy=0.000, audio_variability=0.000, audio_peak=0.000, audio_shift=0.000; visual_energy=0.000, visual_variability=0.000, visual_peak=0.000, visual_shift=0.000'
    keys = [
        'audio_energy', 'audio_variability', 'audio_peak', 'audio_shift',
        'visual_energy', 'visual_variability', 'visual_peak', 'visual_shift',
    ]
    return ', '.join('{}={:.3f}'.format(k, float(stats.get(k, 0.0))) for k in keys)


def make_user_prompt(item, labels):
    multimodal_stats = item.get('multimodal_stats', {})
    return (
        'Choose exactly one emotion label from: {labels}.\n'
        'Return only one JSON object with these keys:\n'
        'label, confidence, rationale, appraisal, modality_hint.\n'
        'appraisal must contain valence, arousal, dominance, certainty in [-1, 1].\n'
        'modality_hint must contain text, audio, visual weights in [0, 1] that sum to 1.\n'
        'Use the retrieved training examples to calibrate label boundaries. Use the current '
        'audio/visual statistics when assigning modality_hint, and use VAD to separate confusing '
        'pairs such as happy/excited, angry/frustrated, and neutral/sad. '
        'Do not use markdown or extra text.\n\n'
        'Current utterance multimodal statistics:\n{stats}\n\n'
        '{prompt}'
    ).format(labels=', '.join(labels), stats=format_stats(multimodal_stats), prompt=item['prompt'])


def build_messages(item, labels):
    return [
        {
            'role': 'system',
            'content': (
                'You are an expert at multimodal emotion recognition. '
                'Reason over dialogue context, retrieved examples, VAD appraisal, and modality reliability.'
            ),
        },
        {
            'role': 'user',
            'content': make_user_prompt(item, labels),
        },
    ]


def parse_response(text, labels, default_confidence):
    data = extract_json(text)
    if not isinstance(data, dict):
        data = {}

    label_value = data.get('label', data.get('emotion', data.get('prediction')))
    if label_value is None:
        label_value = extract_key_value(text, 'label')
    label = normalise_label(label_value, labels)
    if label is None:
        label = normalise_label(text, labels)

    confidence_value = data.get('confidence', data.get('score'))
    if confidence_value is None:
        confidence_value = extract_key_value(text, 'confidence')
    confidence = min(max(safe_float(confidence_value, default_confidence), 0.0), 1.0)
    rationale = data.get('rationale', data.get('reason', ''))
    if not rationale:
        rationale = extract_key_value(text, 'rationale') or ''

    appraisal_prior = fallback_appraisal(label, confidence)
    appraisal = data.get('appraisal', {})
    if not isinstance(appraisal, dict):
        appraisal = {}
    appraisal = {
        'valence': safe_float(appraisal.get('valence', data.get('valence', appraisal_prior['valence']))),
        'arousal': safe_float(appraisal.get('arousal', data.get('arousal', appraisal_prior['arousal']))),
        'dominance': safe_float(appraisal.get('dominance', data.get('dominance', appraisal_prior['dominance']))),
        'certainty': safe_float(appraisal.get('certainty', data.get('certainty', appraisal_prior['certainty']))),
    }
    if appraisal['valence'] == 0.0 and appraisal['arousal'] == 0.0 and appraisal['dominance'] == 0.0 and label in APPRAISAL_PRIORS:
        appraisal.update(appraisal_prior)

    modality_prior = fallback_modality_hint(label)
    modality_hint = data.get('modality_hint', {})
    if not isinstance(modality_hint, dict):
        modality_hint = {}
    modality_hint = {
        'text': max(safe_float(modality_hint.get('text', data.get('text_weight', modality_prior['text']))), 0.0),
        'audio': max(safe_float(modality_hint.get('audio', data.get('audio_weight', modality_prior['audio']))), 0.0),
        'visual': max(safe_float(modality_hint.get('visual', data.get('visual_weight', modality_prior['visual']))), 0.0),
    }
    total_hint = sum(modality_hint.values())
    if total_hint <= 0:
        modality_hint = modality_prior
    else:
        modality_hint = dict((k, v / total_hint) for k, v in modality_hint.items())
    return label, confidence, rationale, appraisal, modality_hint


def build_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not hasattr(torch, 'float8_e4m3fn'):
        torch.float8_e4m3fn = torch.float16
    if not hasattr(torch, 'float8_e5m2'):
        torch.float8_e5m2 = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.cpu:
        dtype = torch.float32
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    elif args.dtype == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32
    model_kwargs = {
        'torch_dtype': dtype,
        'low_cpu_mem_usage': True,
        'local_files_only': True,
        'trust_remote_code': True,
    }
    if args.device_map:
        model_kwargs['device_map'] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if torch.cuda.is_available() and not args.cpu and not args.device_map:
        model = model.cuda()
    model.eval()
    return tokenizer, model


def model_input_device(model):
    import torch

    if hasattr(model, 'hf_device_map'):
        for device in model.hf_device_map.values():
            device = str(device)
            if device not in ('cpu', 'disk'):
                if device.isdigit():
                    device = 'cuda:{}'.format(device)
                return torch.device(device)
    return next(model.parameters()).device


def build_prompt(tokenizer, item, labels):
    messages = build_messages(item, labels)
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]['content'] + '\n\n' + messages[1]['content']


def generate_batch(tokenizer, model, items, labels, args):
    import torch

    prompts = [build_prompt(tokenizer, item, labels) for item in items]
    old_padding_side = getattr(tokenizer, 'padding_side', 'right')
    tokenizer.padding_side = 'left'
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        truncation=True,
        max_length=args.max_input_tokens,
        padding=True,
    )
    tokenizer.padding_side = old_padding_side
    device = model_input_device(model)
    if device.type == 'cuda':
        inputs = dict((k, v.to(device)) for k, v in inputs.items())

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else 1.0,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs['input_ids'].shape[-1]
    responses = []
    for output in outputs:
        generated = output[prompt_length:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return responses


def generate_one(tokenizer, model, item, labels, args):
    return generate_batch(tokenizer, model, [item], labels, args)[0]


def load_raw_dataset(dataset, path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1') if dataset == 'IEMOCAP' else pickle.load(f)

    if dataset == 'MELD':
        video_ids, video_speakers, video_labels, video_text, roberta2, roberta3, roberta4, \
            video_audio, video_visual, video_sentence, train_vid, test_vid, _ = data
    else:
        video_ids, video_speakers, video_labels, video_text, roberta2, roberta3, roberta4, \
            video_audio, video_visual, video_sentence, train_vid, test_vid = data

    return video_speakers, video_labels, video_sentence, video_audio, video_visual, train_vid, test_vid


def safe_array(value):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def utterance_stats(sequence, turn_id):
    current = safe_array(sequence[turn_id])
    previous = safe_array(sequence[turn_id - 1]) if turn_id > 0 else np.zeros_like(current)
    if previous.shape != current.shape:
        previous = np.zeros_like(current)
    return [
        float(np.mean(np.abs(current))),
        float(np.std(current)),
        float(np.max(np.abs(current))) if current.size > 0 else 0.0,
        float(np.mean(np.abs(current - previous))),
    ]


def resolve_vid_key(container, vid):
    if vid in container:
        return vid
    vid_str = str(vid)
    if vid_str in container:
        return vid_str
    try:
        vid_int = int(vid)
        if vid_int in container:
            return vid_int
    except (TypeError, ValueError):
        pass
    raise KeyError(vid)


def multimodal_stats(audio, visual, vid, turn_id):
    audio_vid = resolve_vid_key(audio, vid)
    visual_vid = resolve_vid_key(visual, vid)
    audio_values = utterance_stats(audio[audio_vid], turn_id)
    visual_values = utterance_stats(visual[visual_vid], turn_id)
    keys = [
        'audio_energy', 'audio_variability', 'audio_peak', 'audio_shift',
        'visual_energy', 'visual_variability', 'visual_peak', 'visual_shift',
    ]
    return dict((key, value) for key, value in zip(keys, audio_values + visual_values))


def stats_to_vector(stats):
    return np.asarray([
        stats.get('audio_energy', 0.0),
        stats.get('audio_variability', 0.0),
        stats.get('audio_peak', 0.0),
        stats.get('audio_shift', 0.0),
        stats.get('visual_energy', 0.0),
        stats.get('visual_variability', 0.0),
        stats.get('visual_peak', 0.0),
        stats.get('visual_shift', 0.0),
    ], dtype=np.float32)


def build_context(sentences, speakers, turn_id, window):
    start = max(0, turn_id - window)
    lines = []
    for idx in range(start, turn_id + 1):
        lines.append('{}: {}'.format(speaker_name(speakers[idx]), sentences[idx]))
    return '\n'.join(lines)


def normalise_stat_matrix(matrix):
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normed = (matrix - mean) / std
    length = np.linalg.norm(normed, axis=1, keepdims=True)
    length = np.where(length < 1e-6, 1.0, length)
    return normed / length, mean, std


def normalise_stat_vector(vector, mean, std):
    normed = (vector.reshape(1, -1) - mean) / std
    length = np.linalg.norm(normed, axis=1, keepdims=True)
    length = np.where(length < 1e-6, 1.0, length)
    return normed / length


def build_retrieval_index(dataset, data_path, window):
    labels = EMOTION_LABELS[dataset]
    speakers, gold_labels, sentences, audio, visual, train_vid, test_vid = load_raw_dataset(dataset, data_path)
    records, corpus, stat_vectors = [], [], []

    for vid in train_vid:
        for turn_id, label_id in enumerate(gold_labels[vid]):
            context = build_context(sentences[vid], speakers[vid], turn_id, window)
            stats = multimodal_stats(audio, visual, vid, turn_id)
            record = {
                'vid': str(vid),
                'turn_id': turn_id,
                'label': labels[int(label_id)],
                'utterance': sentences[vid][turn_id],
                'context': context,
                'multimodal_stats': stats,
            }
            records.append(record)
            corpus.append(context)
            stat_vectors.append(stats_to_vector(stats))

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform(corpus)
    stat_matrix, stat_mean, stat_std = normalise_stat_matrix(np.vstack(stat_vectors))
    raw_data = {
        'speakers': speakers,
        'sentences': sentences,
        'audio': audio,
        'visual': visual,
    }
    return vectorizer, matrix, stat_matrix, stat_mean, stat_std, records, raw_data


def load_prompts(path, limit, done):
    prompts = []
    if limit == 0:
        return prompts
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            key = (str(item['vid']), int(item['turn_id']))
            if key in done:
                continue
            prompts.append(item)
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def attach_current_stats(item, raw_data):
    item = dict(item)
    vid = item['vid']
    turn_id = int(item['turn_id'])
    item['multimodal_stats'] = multimodal_stats(raw_data['audio'], raw_data['visual'], vid, turn_id)
    return item


def retrieve_examples(item, vectorizer, matrix, stat_matrix, stat_mean, stat_std, records, k, text_weight):
    query = item.get('prompt', item.get('utterance', ''))
    query_vec = vectorizer.transform([query])
    text_scores = cosine_similarity(query_vec, matrix).ravel()
    stat_vec = normalise_stat_vector(stats_to_vector(item.get('multimodal_stats', {})), stat_mean, stat_std)
    multimodal_scores = np.matmul(stat_matrix, stat_vec.reshape(-1))
    text_weight = min(max(float(text_weight), 0.0), 1.0)
    scores = text_weight * text_scores + (1.0 - text_weight) * multimodal_scores
    order = scores.argsort()[::-1]
    examples = []
    current_key = (str(item['vid']), int(item['turn_id']))

    for idx in order:
        record = records[int(idx)]
        if (record['vid'], int(record['turn_id'])) == current_key:
            continue
        examples.append({
            'score': float(scores[idx]),
            'text_score': float(text_scores[idx]),
            'multimodal_score': float(multimodal_scores[idx]),
            'vid': record['vid'],
            'turn_id': record['turn_id'],
            'label': record['label'],
            'context': record['context'],
            'multimodal_stats': record['multimodal_stats'],
        })
        if len(examples) >= k:
            break
    return examples


def retrieval_quality(examples, predicted_label):
    if len(examples) == 0:
        return {
            'top_score': 0.0,
            'mean_score': 0.0,
            'label_agreement': 0.0,
            'support': 0,
            'quality': 0.0,
        }

    scores = [float(ex.get('score', 0.0)) for ex in examples]
    text_scores = [float(ex.get('text_score', 0.0)) for ex in examples]
    multimodal_scores = [float(ex.get('multimodal_score', 0.0)) for ex in examples]
    agreements = [
        1.0 if predicted_label is not None and str(ex.get('label', '')).lower() == predicted_label else 0.0
        for ex in examples
    ]
    top_score = max(scores)
    mean_score = sum(scores) / len(scores)
    label_agreement = sum(agreements) / len(agreements)
    multimodal_mean = sum(multimodal_scores) / len(multimodal_scores)
    text_mean = sum(text_scores) / len(text_scores)
    quality = 0.35 * top_score + 0.20 * mean_score + 0.20 * max(multimodal_mean, 0.0) + 0.25 * label_agreement
    return {
        'top_score': top_score,
        'mean_score': mean_score,
        'text_score': text_mean,
        'multimodal_score': multimodal_mean,
        'label_agreement': label_agreement,
        'support': len(examples),
        'quality': min(max(quality, 0.0), 1.0),
    }


def build_rag_prompt(item, examples):
    demo_lines = []
    for idx, ex in enumerate(examples, 1):
        demo_lines.append(
            'Example {} (gold label: {}):\n{}\nmultimodal statistics: {}'.format(
                idx, ex['label'], ex['context'], format_stats(ex.get('multimodal_stats', {})))
        )

    return (
        'Retrieved labeled examples from the training set:\n'
        '{}\n\n'
        'Use the examples to calibrate label boundaries and modality reliability. '
        'Do not copy labels blindly; judge the current utterance using both dialogue text and multimodal statistics.\n\n'
        '{}'
    ).format('\n\n'.join(demo_lines), item['prompt'])


def main(args):
    labels = EMOTION_LABELS[args.Dataset]
    data_path = args.data_path or DATA_PATHS[args.Dataset]
    done = load_done(args.output) if args.resume else set()
    prompts = load_prompts(args.prompts, args.limit, done)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print('building RAG index from training split...')
    vectorizer, matrix, stat_matrix, stat_mean, stat_std, records, raw_data = build_retrieval_index(
        args.Dataset, data_path, args.context_window)
    print('indexed {} training utterances'.format(len(records)))
    if len(prompts) == 0:
        print('no prompts to generate')
        return

    tokenizer, model = build_model(args)
    mode = 'a' if args.resume else 'w'
    started = time.time()

    with open(args.output, mode) as f:
        for start in range(0, len(prompts), args.generation_batch_size):
            batch = prompts[start:start + args.generation_batch_size]
            prepared = []
            for item in batch:
                item = attach_current_stats(item, raw_data)
                examples = retrieve_examples(
                    item, vectorizer, matrix, stat_matrix, stat_mean, stat_std,
                    records, args.rag_k, args.text_rag_weight)
                rag_item = dict(item)
                rag_item['prompt'] = build_rag_prompt(item, examples)
                prepared.append((item, examples, rag_item))

            responses = generate_batch(tokenizer, model, [entry[2] for entry in prepared], labels, args)

            for offset, (item, examples, rag_item) in enumerate(prepared):
                idx = start + offset + 1
                response = responses[offset]
                label, confidence, rationale, appraisal, modality_hint = parse_response(response, labels, args.default_confidence)
                rag_quality = retrieval_quality(examples, label)
                record = {
                    'vid': item['vid'],
                    'turn_id': item['turn_id'],
                    'label': label,
                    'confidence': confidence,
                    'rationale': rationale,
                    'appraisal': appraisal,
                    'modality_hint': modality_hint,
                    'multimodal_stats': item.get('multimodal_stats', {}),
                    'reasoning_quality': rag_quality,
                    'retrieved_examples': examples,
                    'raw_response': response,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()

            done_count = min(start + len(batch), len(prompts))
            if done_count % args.log_every == 0 or done_count == len(prompts):
                elapsed = round(time.time() - started, 2)
                print('generated {}/{} RAG records, elapsed {} sec'.format(done_count, len(prompts), elapsed))

    print('saved {} new RAG reasoning records to {}'.format(len(prompts), args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset', default='IEMOCAP', choices=['IEMOCAP', 'MELD'])
    parser.add_argument('--data-path', default='', help='path to multimodal feature pkl')
    parser.add_argument('--prompts', default='data/iemocap_llm_prompts.jsonl')
    parser.add_argument('--output', default='data/iemocap_llm_reasoning_mmrag.jsonl')
    parser.add_argument('--model-path', default='/data/LLM/Qwen2.5-7B-Instruct')
    parser.add_argument('--rag-k', type=int, default=5)
    parser.add_argument('--context-window', type=int, default=3)
    parser.add_argument('--text-rag-weight', type=float, default=0.7, help='weight of text retrieval score; remaining weight uses audio/visual statistics')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--device-map', default='', help='device map for large LLMs, e.g. auto for multi-GPU inference')
    parser.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--generation-batch-size', type=int, default=1, help='number of prompts generated together; increase for faster throughput when memory allows')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--default-confidence', type=float, default=0.85)
    parser.add_argument('--log-every', type=int, default=50)
    args = parser.parse_args()
    main(args)

import csv
import json
import os
import pickle

import torch


EMOTION_LABELS = {
    'IEMOCAP': ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
    'MELD': ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
}

APPRAISAL_PRIORS = {
    'happy': [0.70, 0.45, 0.35, 0.70],
    'joy': [0.85, 0.65, 0.45, 0.75],
    'joyful': [0.85, 0.65, 0.45, 0.75],
    'excited': [0.85, 0.90, 0.50, 0.75],
    'sad': [-0.80, -0.45, -0.55, 0.65],
    'sadness': [-0.80, -0.45, -0.55, 0.65],
    'neutral': [0.00, 0.00, 0.00, 0.60],
    'angry': [-0.75, 0.85, 0.65, 0.75],
    'anger': [-0.75, 0.85, 0.65, 0.75],
    'frustrated': [-0.65, 0.70, -0.35, 0.70],
    'frustration': [-0.65, 0.70, -0.35, 0.70],
    'fear': [-0.70, 0.80, -0.70, 0.65],
    'surprise': [0.30, 0.85, 0.10, 0.55],
    'disgust': [-0.70, 0.55, 0.20, 0.70],
}

MODALITY_PRIORS = {
    'happy': [0.45, 0.25, 0.30],
    'joy': [0.40, 0.30, 0.30],
    'joyful': [0.40, 0.30, 0.30],
    'excited': [0.25, 0.55, 0.20],
    'sad': [0.45, 0.35, 0.20],
    'sadness': [0.45, 0.35, 0.20],
    'neutral': [0.55, 0.25, 0.20],
    'angry': [0.25, 0.55, 0.20],
    'anger': [0.25, 0.55, 0.20],
    'frustrated': [0.35, 0.45, 0.20],
    'frustration': [0.35, 0.45, 0.20],
    'fear': [0.35, 0.45, 0.20],
    'surprise': [0.35, 0.25, 0.40],
    'disgust': [0.35, 0.25, 0.40],
}


LLM_EXTRA_FEATURE_DIM = 12


def cognitive_feature_dim(n_classes):
    return n_classes + LLM_EXTRA_FEATURE_DIM


class LLMReasoningChain(object):
    """Loads offline LLM emotion reasoning outputs and aligns them to a batch."""

    def __init__(self, path, dataset, n_classes, label_confidence=0.85, temperature=1.0):
        self.path = path
        self.dataset = dataset
        self.n_classes = n_classes
        self.label_confidence = label_confidence
        self.temperature = temperature
        self.labels = EMOTION_LABELS.get(dataset, [])
        self.label_to_idx = dict((label.lower(), idx) for idx, label in enumerate(self.labels))
        self.feature_dim = cognitive_feature_dim(n_classes)
        self.cache = {}
        self.enabled = bool(path)

        if self.enabled:
            self.cache = self._load_cache(path)

    def _load_cache(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                'LLM reasoning cache not found: {}. Generate it first, or run without --llm-cache. '
                'You can export prompts with: python export_llm_prompts.py --Dataset {} --output data/{}_llm_prompts.jsonl'.format(
                    path, self.dataset, self.dataset.lower()))

        ext = os.path.splitext(path)[1].lower()
        if ext in ['.pkl', '.pk']:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif ext == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif ext == '.jsonl':
            data = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        elif ext == '.csv':
            with open(path, 'r') as f:
                data = list(csv.DictReader(f))
        else:
            raise ValueError('Unsupported LLM cache format: {}'.format(path))

        if isinstance(data, dict) and self.dataset in data:
            data = data[self.dataset]

        return self._normalise_cache(data)

    def _normalise_cache(self, data):
        cache = {}
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, tuple) and len(key) == 2:
                    vid, turn_id = key
                    cache.setdefault(str(vid), {})[int(turn_id)] = value
                else:
                    cache[str(key)] = value
            return cache

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                vid = item.get('vid', item.get('dialogue_id', item.get('dialog_id')))
                turn_id = item.get('turn_id', item.get('utterance_idx', item.get('index')))
                if vid is None:
                    continue
                if turn_id is None:
                    cache[str(vid)] = item.get('turns', item.get('utterances', item.get('predictions', item)))
                else:
                    cache.setdefault(str(vid), {})[int(turn_id)] = item
        return cache

    def _distribution_from_label(self, label, confidence=None):
        if label is None:
            return None
        idx = self.label_to_idx.get(str(label).lower())
        if idx is None:
            try:
                idx = int(label)
            except (TypeError, ValueError):
                return None
        if idx < 0 or idx >= self.n_classes:
            return None

        try:
            confidence = self.label_confidence if confidence is None else float(confidence)
        except (TypeError, ValueError):
            confidence = self.label_confidence
        confidence = min(max(confidence, 0.0), 1.0)
        dist = torch.ones(self.n_classes, dtype=torch.float) * ((1.0 - confidence) / max(self.n_classes - 1, 1))
        dist[idx] = confidence
        return dist

    def _distribution_from_probs(self, probs):
        if probs is None:
            return None
        if isinstance(probs, dict):
            dense = [0.0] * self.n_classes
            for label, value in probs.items():
                idx = self.label_to_idx.get(str(label).lower())
                if idx is None:
                    try:
                        idx = int(label)
                    except (TypeError, ValueError):
                        continue
                if 0 <= idx < self.n_classes:
                    try:
                        dense[idx] = float(value)
                    except (TypeError, ValueError):
                        continue
            probs = dense

        if len(probs) != self.n_classes:
            return None
        try:
            dist = torch.FloatTensor([float(x) for x in probs])
        except (TypeError, ValueError):
            return None
        dist = torch.clamp(dist, min=0.0)
        if dist.sum().item() <= 0:
            return None
        if self.temperature != 1.0:
            dist = torch.pow(dist, 1.0 / self.temperature)
        return dist / dist.sum()

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _confidence_from_item(self, item):
        if isinstance(item, dict):
            return min(max(self._safe_float(item.get('confidence', item.get('score')), self.label_confidence), 0.0), 1.0)
        return self.label_confidence

    def _rag_quality_from_item(self, item, label):
        if not isinstance(item, dict):
            return self.label_confidence * 0.75

        quality = item.get('reasoning_quality', item.get('rag_quality'))
        if isinstance(quality, dict):
            if 'quality' in quality:
                quality = self._safe_float(quality.get('quality'), 0.0)
            else:
                values = [
                    self._safe_float(quality.get('top_score'), 0.0),
                    self._safe_float(quality.get('mean_score'), 0.0),
                    self._safe_float(quality.get('label_agreement'), 0.0),
                ]
                quality = 0.45 * values[0] + 0.30 * values[1] + 0.25 * values[2]
        elif quality is not None:
            quality = self._safe_float(quality, 0.0)
        else:
            examples = item.get('retrieved_examples', [])
            if isinstance(examples, list) and len(examples) > 0:
                scores, agreements = [], []
                for ex in examples:
                    if not isinstance(ex, dict):
                        continue
                    scores.append(self._safe_float(ex.get('score'), 0.0))
                    ex_label = str(ex.get('label', '')).lower()
                    agreements.append(1.0 if label is not None and ex_label == label else 0.0)
                top_score = max(scores) if scores else 0.0
                mean_score = sum(scores) / len(scores) if scores else 0.0
                agreement = sum(agreements) / len(agreements) if agreements else 0.0
                quality = 0.45 * top_score + 0.30 * mean_score + 0.25 * agreement
            else:
                quality = self._confidence_from_item(item) * 0.75
        return min(max(float(quality), 0.0), 1.0)

    def _label_from_item(self, item):
        if item is None:
            return None
        if not isinstance(item, dict):
            return str(item).lower()
        label = item.get('label', item.get('emotion', item.get('prediction')))
        return str(label).lower() if label is not None else None

    def _item_to_distribution(self, item):
        if item is None:
            return None
        if isinstance(item, (list, tuple)):
            return self._distribution_from_probs(item)
        if not isinstance(item, dict):
            return self._distribution_from_label(item)

        probs = item.get('probs', item.get('prob', item.get('distribution', item.get('scores'))))
        if probs is None:
            prob_cols = {}
            for label in self.labels:
                if label in item:
                    prob_cols[label] = item[label]
            probs = prob_cols if prob_cols else None

        dist = self._distribution_from_probs(probs)
        if dist is not None:
            return dist

        label = item.get('label', item.get('emotion', item.get('prediction')))
        confidence = item.get('confidence', item.get('score'))
        return self._distribution_from_label(label, confidence)

    def _appraisal_from_item(self, item):
        appraisal = item.get('appraisal', {}) if isinstance(item, dict) else {}
        label = self._label_from_item(item)
        prior = APPRAISAL_PRIORS.get(label, [0.0, 0.0, 0.0, self._confidence_from_item(item)])
        values = [
            appraisal.get('valence', item.get('valence', prior[0]) if isinstance(item, dict) else prior[0]),
            appraisal.get('arousal', item.get('arousal', prior[1]) if isinstance(item, dict) else prior[1]),
            appraisal.get('dominance', item.get('dominance', prior[2]) if isinstance(item, dict) else prior[2]),
            appraisal.get('certainty', item.get('certainty', prior[3]) if isinstance(item, dict) else prior[3]),
        ]
        values = [self._safe_float(v, prior[i]) for i, v in enumerate(values)]
        if values[0] == 0.0 and values[1] == 0.0 and values[2] == 0.0 and label in APPRAISAL_PRIORS:
            values[:3] = APPRAISAL_PRIORS[label][:3]
        return torch.FloatTensor(values)

    def _vad_guidance_from_appraisal(self, appraisal):
        valence = min(max(float(appraisal[0]), -1.0), 1.0)
        arousal = min(max(float(appraisal[1]), -1.0), 1.0)
        dominance = min(max(float(appraisal[2]), -1.0), 1.0)
        positive_activation = max(valence, 0.0) * arousal
        negative_control_axis = max(-valence, 0.0) * max(arousal, 0.0) * dominance
        negative_low_arousal = max(-valence, 0.0) * max(-arousal, 0.0)
        return torch.FloatTensor([positive_activation, negative_control_axis, negative_low_arousal])

    def _modality_hint_from_item(self, item):
        hint = item.get('modality_hint', {}) if isinstance(item, dict) else {}
        label = self._label_from_item(item)
        prior = MODALITY_PRIORS.get(label, [1.0, 1.0, 1.0])
        values = [
            hint.get('text', item.get('text_weight', prior[0]) if isinstance(item, dict) else prior[0]),
            hint.get('audio', item.get('audio_weight', prior[1]) if isinstance(item, dict) else prior[1]),
            hint.get('visual', item.get('visual_weight', prior[2]) if isinstance(item, dict) else prior[2]),
        ]
        hint = torch.FloatTensor([max(self._safe_float(v, prior[i]), 0.0) for i, v in enumerate(values)])
        if hint.sum().item() <= 0:
            hint = torch.ones(3, dtype=torch.float)
        return hint / hint.sum()

    def _item_to_features(self, item):
        dist = self._item_to_distribution(item)
        if dist is None:
            return None
        label = self._label_from_item(item)
        confidence = torch.FloatTensor([self._confidence_from_item(item)])
        appraisal = self._appraisal_from_item(item)
        modality_hint = self._modality_hint_from_item(item)
        rag_quality = torch.FloatTensor([self._rag_quality_from_item(item, label)])
        vad_guidance = self._vad_guidance_from_appraisal(appraisal)
        return torch.cat([dist, appraisal, modality_hint, confidence, rag_quality, vad_guidance], dim=0)

    def _raw_turn_item(self, vid, turn_id):
        dialogue = self.cache.get(str(vid))
        if dialogue is None:
            return None
        if isinstance(dialogue, dict):
            item = dialogue.get(turn_id, dialogue.get(str(turn_id)))
        elif isinstance(dialogue, list) and turn_id < len(dialogue):
            item = dialogue[turn_id]
        else:
            item = None
        return item

    def _get_turn(self, vid, turn_id):
        item = self._raw_turn_item(vid, turn_id)
        return self._item_to_distribution(item)

    def batch_cognitive_features(self, dialogue_ids, lengths, max_len, device):
        if not self.enabled:
            return None, None

        batch_size = len(dialogue_ids)
        features = torch.zeros(batch_size, max_len, self.feature_dim, dtype=torch.float)
        mask = torch.zeros(batch_size, max_len, dtype=torch.float)
        matched = 0

        for batch_id, vid in enumerate(dialogue_ids):
            for turn_id in range(lengths[batch_id]):
                item = self._raw_turn_item(vid, turn_id)
                feat = self._item_to_features(item)
                if feat is None:
                    continue
                features[batch_id, turn_id] = feat
                mask[batch_id, turn_id] = 1.0
                matched += 1

        if matched == 0:
            return None, None
        return features.to(device), mask.to(device)

    def coverage(self, dataloader):
        total, matched = 0, 0
        for data in dataloader:
            dialogue_ids = data[-1]
            umask = data[4]
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            for batch_id, vid in enumerate(dialogue_ids):
                for turn_id in range(lengths[batch_id]):
                    total += 1
                    matched += 1 if self._get_turn(vid, turn_id) is not None else 0
        return matched, total

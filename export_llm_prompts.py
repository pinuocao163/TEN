import argparse
import json
import pickle

from llm_reasoning import EMOTION_LABELS


DATA_PATHS = {
    'IEMOCAP': '/data/zzb/BaseLine/SDT/data/iemocap_multimodal_features.pkl',
    'MELD': '/data/zzb/BaseLine/SDT/data/meld_multimodal_features.pkl',
}


def load_dataset(dataset, path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1') if dataset == 'IEMOCAP' else pickle.load(f)

    if dataset == 'MELD':
        video_ids, video_speakers, video_labels, video_text, roberta2, roberta3, roberta4, \
            video_audio, video_visual, video_sentence, train_vid, test_vid, _ = data
    else:
        video_ids, video_speakers, video_labels, video_text, roberta2, roberta3, roberta4, \
            video_audio, video_visual, video_sentence, train_vid, test_vid = data

    return video_speakers, video_sentence, train_vid, test_vid


def speaker_name(speaker):
    if isinstance(speaker, (list, tuple)):
        if len(speaker) == 2:
            return 'M' if speaker[0] == 1 else 'F'
        if len(speaker) > 0:
            return 'Speaker{}'.format(max(range(len(speaker)), key=lambda i: speaker[i]))
    return str(speaker)


def build_prompt(dataset, labels, context_lines, current_line):
    return (
        'You are an expert at emotion recognition in conversations.\n'
        'Choose exactly one emotion label from: {}.\n'
        'Use the dialogue context, speaker turns, emotional cues, and pragmatic intent.\n'
        'Return only valid JSON with keys: label, confidence, rationale.\n\n'
        'Dialogue context:\n{}\n\n'
        'Current utterance:\n{}\n'.format(', '.join(labels), '\n'.join(context_lines), current_line)
    )


def export_prompts(dataset, data_path, output, split):
    labels = EMOTION_LABELS[dataset]
    video_speakers, video_sentence, train_vid, test_vid = load_dataset(dataset, data_path)
    if split == 'train':
        vids = train_vid
    elif split == 'test':
        vids = test_vid
    else:
        vids = list(train_vid) + list(test_vid)

    total = 0
    with open(output, 'w') as f:
        for vid in vids:
            sentences = video_sentence[vid]
            speakers = video_speakers[vid]
            context_lines = []
            for turn_id, sentence in enumerate(sentences):
                speaker = speaker_name(speakers[turn_id])
                current_line = '{}: {}'.format(speaker, sentence)
                prompt = build_prompt(dataset, labels, context_lines + [current_line], current_line)
                record = {
                    'vid': str(vid),
                    'turn_id': turn_id,
                    'speaker': speaker,
                    'utterance': sentence,
                    'labels': labels,
                    'prompt': prompt,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                context_lines.append(current_line)
                total += 1
    print('exported {} prompts to {}'.format(total, output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset', default='IEMOCAP', choices=['IEMOCAP', 'MELD'])
    parser.add_argument('--data-path', default='', help='path to multimodal feature pkl')
    parser.add_argument('--output', required=True, help='output jsonl prompt file')
    parser.add_argument('--split', default='all', choices=['train', 'test', 'all'])
    args = parser.parse_args()

    data_path = args.data_path or DATA_PATHS[args.Dataset]
    export_prompts(args.Dataset, data_path, args.output, args.split)

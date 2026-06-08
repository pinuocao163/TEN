import argparse
import csv
import os
import re
from statistics import mean, stdev


START_RE = re.compile(r'Dataset=(?P<dataset>\w+)\s+Setting=(?P<setting>\S+)\s+Run=(?P<run>\d+)/(?P<runs>\d+)\s+Start=')
COVERAGE_RE = re.compile(r'LLM reasoning cache coverage: train (?P<tr_hit>\d+)/(?P<tr_total>\d+), test (?P<te_hit>\d+)/(?P<te_total>\d+)')
BEST_ACC_RE = re.compile(r'Best Acc:\s*(?P<value>[-+]?\d+(?:\.\d+)?)')
FSCORE_RE = re.compile(r'F-Score:\s*(?P<value>[-+]?\d+(?:\.\d+)?)')
EPOCH_RE = re.compile(r'F-Score-index:\s*(?P<value>\d+)')


def parse_log(path):
    rows = []
    current = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = START_RE.search(line)
            if m:
                if current is not None:
                    rows.append(current)
                current = {
                    'log_file': path,
                    'dataset': m.group('dataset'),
                    'setting': m.group('setting'),
                    'run': int(m.group('run')),
                    'runs': int(m.group('runs')),
                    'train_cache_hit': '',
                    'train_cache_total': '',
                    'test_cache_hit': '',
                    'test_cache_total': '',
                    'best_acc': '',
                    'best_f1': '',
                    'best_epoch': '',
                }
                continue
            if current is None:
                continue
            m = COVERAGE_RE.search(line)
            if m:
                current['train_cache_hit'] = int(m.group('tr_hit'))
                current['train_cache_total'] = int(m.group('tr_total'))
                current['test_cache_hit'] = int(m.group('te_hit'))
                current['test_cache_total'] = int(m.group('te_total'))
                continue
            m = BEST_ACC_RE.search(line)
            if m:
                current['best_acc'] = float(m.group('value'))
                continue
            m = FSCORE_RE.search(line)
            if m:
                current['best_f1'] = float(m.group('value'))
                continue
            m = EPOCH_RE.search(line)
            if m:
                current['best_epoch'] = int(m.group('value'))
                continue
    if current is not None:
        rows.append(current)
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row['dataset'], row['setting'])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (dataset, setting), items in sorted(grouped.items()):
        f1_values = [x['best_f1'] for x in items if isinstance(x.get('best_f1'), float)]
        acc_values = [x['best_acc'] for x in items if isinstance(x.get('best_acc'), float)]
        if not f1_values:
            continue
        summary.append({
            'dataset': dataset,
            'setting': setting,
            'runs': len(f1_values),
            'mean_f1': round(mean(f1_values), 4),
            'std_f1': round(stdev(f1_values), 4) if len(f1_values) > 1 else 0.0,
            'max_f1': round(max(f1_values), 4),
            'mean_acc': round(mean(acc_values), 4) if acc_values else '',
            'std_acc': round(stdev(acc_values), 4) if len(acc_values) > 1 else 0.0,
            'max_acc': round(max(acc_values), 4) if acc_values else '',
        })
    return summary


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+', help='parameter sensitivity log files')
    parser.add_argument('--output', default='', help='CSV path for per-run rows')
    parser.add_argument('--summary-output', default='', help='CSV path for grouped summary')
    args = parser.parse_args()

    rows = []
    for log in args.logs:
        rows.extend(parse_log(log))

    run_fields = [
        'log_file', 'dataset', 'setting', 'run', 'runs',
        'train_cache_hit', 'train_cache_total', 'test_cache_hit', 'test_cache_total',
        'best_acc', 'best_f1', 'best_epoch',
    ]
    summary_fields = [
        'dataset', 'setting', 'runs', 'mean_f1', 'std_f1', 'max_f1',
        'mean_acc', 'std_acc', 'max_acc',
    ]

    if args.output:
        write_csv(args.output, rows, run_fields)
    if args.summary_output:
        write_csv(args.summary_output, summarize(rows), summary_fields)

    print('Parsed {} runs from {} log file(s).'.format(len(rows), len(args.logs)))
    if not args.output and not args.summary_output:
        for row in summarize(rows):
            print(row)


if __name__ == '__main__':
    main()

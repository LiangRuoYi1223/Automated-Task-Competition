import argparse
import json
from pathlib import Path
import sys
from .data import load_dataset
from .runner import (VARIANTS, backend_module, compare_reports, read_config,
                     read_json, run_prediction, score_prediction, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Four trained top-face models + preserved nontrained baseline')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('variants', help='list supported variants without importing training frameworks')
    inspect = commands.add_parser('inspect-data', help='audit COCO scene/content isolation')
    inspect.add_argument('--dataset', required=True)
    inspect.add_argument('--split', choices=('train','val','test'), default='val')
    for name in ('train', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('--dataset', required=True)
        command.add_argument('--config', required=True)
        command.add_argument('--output', required=True)
        if name == 'predict':
            command.add_argument('--split', choices=('val','test'), default='val')
            command.add_argument('--warmup', type=int, default=2)
    score = commands.add_parser('score')
    score.add_argument('--dataset', required=True)
    score.add_argument('--predictions', required=True)
    score.add_argument('--policy')
    score.add_argument('--output', required=True)
    compare = commands.add_parser('compare')
    compare.add_argument('--reports', nargs='+', required=True)
    compare.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'variants':
            print('\n'.join(VARIANTS))
        elif args.command == 'inspect-data':
            dataset = load_dataset(args.dataset, args.split)
            print(json.dumps({'images': len(dataset.records), 'instances': sum(len(r.instances) for r in dataset.records),
                              'classes': dataset.classes, 'fingerprint': dataset.fingerprint}, ensure_ascii=False))
        elif args.command == 'train':
            config = read_config(args.config)
            if Path(args.output).exists():
                raise FileExistsError('training output already exists: ' + args.output)
            module = backend_module(config)
            if not hasattr(module, 'train'):
                raise ValueError('the nontrained baseline has no training stage')
            result = module.train(args.dataset, args.output, config)
            print(json.dumps(result, ensure_ascii=False, default=str))
        elif args.command == 'predict':
            result = run_prediction(args.dataset, args.split, read_config(args.config), args.output, args.warmup)
            print('Saved {} complete frames to {}'.format(len(result['frames']), args.output))
        elif args.command == 'score':
            result = score_prediction(args.dataset, args.predictions, read_json(args.policy) if args.policy else None)
            write_json(args.output, result)
            print(json.dumps(result['metrics'], ensure_ascii=False, allow_nan=False))
        else:
            result = compare_reports([read_json(p) for p in args.reports])
            write_json(args.output, result)
            for row in result['ranking']:
                print('{}: score={:.2f}, geometry={:.2f}, p95={:.2f}ms'.format(
                    row['variant'], row['score'], row['geometry_score'], row['latency_ms']['p95']))
            print('Offline candidate: ' + str(result['recommended_offline_candidate']))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as error:
        print('{}: {}'.format(type(error).__name__, error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

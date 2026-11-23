"""Generate one source-model experiment, then evaluate all four retrieval models."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from paths import ROOT, resolve

MODELS = ('clip_vit', 'clip_rn101', 'albef', 'tcl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', choices=MODELS, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--eval-gpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--run-dir', type=Path, help='Parent directory, relative to this project')
    parser.add_argument('--targets', choices=MODELS, nargs='+', default=list(MODELS))
    parser.add_argument('--tf32', choices=('on', 'off'), default='on')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.gpu.isdecimal() or (args.eval_gpu is not None and not args.eval_gpu.isdecimal()):
        parser.error('GPU indices must be nonnegative integers')
    run_dir = resolve(args.run_dir or f'runs/retrieval_{args.source_model}_mixed11_seed{args.seed}_b{args.batch_size}')
    seed_dir = run_dir / f'seed_{args.seed}'
    python = [sys.executable, '-u', '-B']
    train = python + [str(ROOT / 'train.py'), '--source-model', args.source_model,
                      '--run-dir', str(seed_dir), '--gpu', args.gpu,
                      '--seed', str(args.seed), '--batch-size', str(args.batch_size), '--tf32', args.tf32]
    if args.resume:
        train.append('--resume')
    if args.max_images is not None:
        train += ['--max-images', str(args.max_images)]
    evaluations = {target: python + [str(ROOT / 'evaluate.py'), '--run-dir', str(seed_dir),
                                    '--targets', target, '--gpu', args.eval_gpu or args.gpu,
                                    '--tf32', args.tf32] for target in args.targets}
    if args.dry_run:
        print(json.dumps(dict(output=str(run_dir.relative_to(ROOT)),
                              generation=None if args.eval_only else train, evaluations=evaluations), indent=2))
        return
    logs = run_dir / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    if not args.worker and not args.foreground:
        pid_file = logs / 'coordinator.pid'
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Coordinator may still be active: {pid_file}')
        with (logs / 'coordinator.log').open('a') as stream:
            process = subprocess.Popen(python + [str(ROOT / 'run.py'), *sys.argv[1:], '--worker'],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        pid_file.write_text(str(process.pid) + '\n')
        print(f'PID: {process.pid}\nLog: {logs / "coordinator.log"}')
        return
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    if not args.eval_only:
        with (logs / f'seed_{args.seed}_train.log').open('a') as stream:
            subprocess.run(train, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for target, command in evaluations.items():
        print(f'Evaluating {target}', flush=True)
        with (logs / f'seed_{args.seed}_{target}.log').open('a') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    rows = []
    for target in args.targets:
        result = json.loads((seed_dir / 'evaluations' / f'{target}.json').read_text())
        if not result['full_flickr30k']:
            print('Subset evaluation complete. ASR requires the full test split.', flush=True)
            return
        rows.append(dict(source_model=args.source_model, target_model=target, white_box=target == args.source_model,
                         TR_R1_ASR=result['tr1_asr'], IR_R1_ASR=result['ir1_asr']))
    (run_dir / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (run_dir / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Completed: ' + str(run_dir), flush=True)


if __name__ == '__main__':
    main()

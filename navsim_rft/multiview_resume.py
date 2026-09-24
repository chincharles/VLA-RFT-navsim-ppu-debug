"""Restore trusted project checkpoints, without overwriting earlier attempts."""
import hashlib
import json
from pathlib import Path
from uuid import uuid4


def prepare(args, modules, optimizer, dataset, rank, world, stage):
    from .checkpoint import load
    out = Path(args.output)
    signature = dict(batch_size=args.batch_size, seed=args.seed,
                     manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                     stats=dataset.stats.record)
    metadata = dict(stage=stage, steps=args.steps, views=list(dataset.EXPECTED_VIEWS),
                    signature=signature)
    if hasattr(args, 'tokenizer'):
        metadata['tokenizer'] = args.tokenizer
    if args.steps < 1 or args.batch_size < 1 or args.save_every < 0:
        raise ValueError('Invalid training budget')
    start = 0
    if args.resume:
        import torch
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        if ck['metadata']['stage'] != stage or ck['metadata']['views'] != metadata['views']:
            raise ValueError('Checkpoint stage/cameras mismatch')
        if len(ck['rng']) != world:
            raise ValueError('Resume requires the original world size')
        if 'signature' in ck['metadata'] and ck['metadata']['signature'] != signature:
            raise ValueError('Resume dataset/statistics/batch/seed mismatch')
        start = int(ck['step'])
        if start > args.steps:
            raise ValueError('--steps is the TOTAL target, less than saved step')
        if 'signature' not in ck['metadata'] and rank == 0:
            print('Legacy checkpoint: use the original data, stats, batch and seed; provenance was not saved.', flush=True)
        del ck
        load(args.resume, modules, optimizer, rank=rank)
        # Prevent an explicit older resume from overwriting later checkpoints.
        if out.exists() and any(start < int(p.stem.split('-')[1]) <= args.steps
                                for p in out.glob('step-*.pt')):
            raise ValueError('Later checkpoints exist; resume latest or use a fresh output')
    elif out.exists():
        raise FileExistsError('Use --resume or a fresh output directory')
    if rank == 0:
        out.mkdir(parents=True, exist_ok=bool(args.resume))
        print(f'{stage}: saved step={start}, target={args.steps}, remaining={args.steps-start}', flush=True)
    # Never append restarted steps to the old metrics: retain its unsaved tail.
    log = out / (f'metrics-resume-{start:06d}-{uuid4().hex[:8]}.jsonl' if args.resume else 'metrics.jsonl')
    if rank == 0:
        print(f'Metrics: {log}', flush=True)
    return start, metadata, log

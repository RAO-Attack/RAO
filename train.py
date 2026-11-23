"""Generate adversarial images and captions for Flickr30K retrieval."""
import argparse
import json
import time
from pathlib import Path

import torch

from common import (ANNOTATION, BERT, DATA, NORMALIZE, SOURCES,
                    attack_configuration, attack_encoders, checked_batch, checked_text_batch,
                    configuration, configure_tf32, dataset, device_for, load_clip, load_fusion,
                    load_text_attacker, relative, resolve, restore_rng, rng_state,
                    save_torch, seed_everything, tokenizer, write_json)
from RAttacker import Attacker, ImageAttacker, rgta_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-model', choices=tuple(SOURCES), default='clip_vit')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tf32', choices=('on', 'off'), default='on')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-images', type=int, help='Limit the number of input images')
    parser.add_argument('--stop-after-batches', type=int, help='Commit this many new batches, then stop')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.stop_after_batches is not None and args.stop_after_batches < 1:
        parser.error('--stop-after-batches must be positive')
    config = attack_configuration()
    device = device_for(args.gpu, args.device, args.threads)
    precision = configure_tf32(args.tf32)
    data = dataset(ANNOTATION, DATA, args.max_images, source=args.source_model)
    text_length = SOURCES[args.source_model][2]
    run_dir = resolve(args.run_dir or f'runs/retrieval_{args.source_model}_mixed11/seed_{args.seed}')
    settings = dict(method='RAO-mixed11', source_model=args.source_model, seed=args.seed,
                    batch_size=args.batch_size, eps=config['eps'], steps=config['steps'],
                    step_size=config['step_size'], exp_num=config['exp_num'],
                    annotation=relative(ANNOTATION), image_root=relative(DATA),
                    bert_dir=relative(BERT),
                    max_images=args.max_images, num_images=len(data), num_texts=len(data.text),
                    text_attack=True, attack_mode='joint', attack_text_length=text_length,
                    image_augmentation=dict(mode='mixed11', **config['mixed11']),
                    text_attack_settings=rgta_settings(), **precision)
    manifest_path = run_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['settings'] != settings:
            raise ValueError('Existing run configuration differs; use a new output directory')
        if not args.resume and manifest['status'] != 'complete':
            raise FileExistsError('Partial run exists; pass --resume')
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f'Output directory is not empty: {run_dir}')
        (run_dir / 'batches').mkdir(parents=True)
        manifest = dict(status='running', settings=settings, batches=[])
        write_json(manifest_path, manifest)
    done, previous_state = 0, None
    for record in manifest['batches']:
        expected = list(range(done, min(done + args.batch_size, len(data))))
        item = checked_batch(run_dir, record, expected, source=args.source_model)
        checked_text_batch(item, record, data)
        previous_state = item['rng']
        done += len(expected)
    if done == len(data):
        manifest['status'] = 'complete'
        write_json(manifest_path, manifest)
        print(f'Generation already complete: {run_dir}', flush=True)
        return
    seed_everything(args.seed)
    model = (load_clip(args.source_model, BERT, device) if args.source_model.startswith('clip_')
             else load_fusion(args.source_model, BERT, device, configuration()))
    tok = tokenizer(BERT)
    image_attacker = ImageAttacker(NORMALIZE, eps=config['eps'], steps=config['steps'],
                                   step_size=config['step_size'])
    image_attacker.configure_mixed_augmentation(**config['mixed11'])
    text_attacker = load_text_attacker(BERT, tok, device, max_length=text_length)
    image_encoder, text_encoder = attack_encoders(model, device)
    attacker = Attacker(model, image_encoder, text_encoder, image_attacker, text_attacker)
    # Keep iterator creation from consuming the attack RNG on resume.
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, num_workers=0,
                                        collate_fn=data.collate_fn, generator=generator)
    if previous_state is not None:
        restore_rng(previous_state)
    started, committed = time.monotonic(), 0
    for batch_index, (images, groups, image_ids, text_id_groups) in enumerate(loader):
        if image_ids[-1] < done:
            continue
        texts = [text for group in groups for text in group]
        txt2img = [i for i, group in enumerate(groups) for _ in group]
        images = images.to(device)
        adversarial, adversarial_texts = attacker.attack91(
            images, texts, txt2img, device=device, max_length=text_length,
            scales=[0.5, 0.75, 1.25, 1.5])
        linf = (adversarial - images).abs().max().item()
        if not torch.isfinite(adversarial).all() or linf > config['eps'] + 1e-6:
            raise RuntimeError('Adversarial image is nonfinite or exceeds its budget')
        filename = f'batches/batch_{batch_index:05d}.pt'
        item = dict(image_ids=image_ids, images=adversarial.detach().cpu().float(), rng=rng_state(),
                    text_ids=[i for ids in text_id_groups for i in ids], adv_texts=adversarial_texts)
        checked_text_batch(item, {}, data)
        save_torch(run_dir / filename, item)
        record = dict(file=filename, image_ids=image_ids,
                      linf=linf, text_ids=item['text_ids'],
                      changed_texts=sum(a != b for a, b in zip(texts, adversarial_texts)))
        manifest['batches'].append(record)
        write_json(manifest_path, manifest)
        committed += 1
        print(f'Generated {image_ids[-1] + 1}/{len(data)}; L_inf={linf:.8f}', flush=True)
        if args.stop_after_batches and committed >= args.stop_after_batches and image_ids[-1] + 1 < len(data):
            print('Stopped after a committed batch; continue with --resume', flush=True)
            return
    manifest['status'] = 'complete'
    manifest['last_session_seconds'] = time.monotonic() - started
    write_json(manifest_path, manifest)
    print(f'Generation complete: {run_dir}', flush=True)


if __name__ == '__main__':
    main()

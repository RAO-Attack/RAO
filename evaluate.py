"""Evaluate saved adversarial images and captions."""
import argparse
import gc
import json
from pathlib import Path

import torch

import evaluation
from common import (BERT, TARGETS, NORMALIZE, checked_batch,
                    checked_text_batch, configuration, configure_tf32, dataset, device_for,
                    load_clip, load_fusion, read_manifest, relative, resolve,
                    seed_everything, target_transform, tokenizer, write_json)
from paths import ANNOTATION


@torch.no_grad()
def evaluate_target(run_dir, manifest, target, args, device):
    settings = manifest['settings']
    source = settings['source_model']
    config = configuration()
    data = dataset(resolve(settings['annotation']), resolve(settings['image_root']),
                   settings['max_images'], source=source)
    full = (settings['max_images'] is None and len(data) == 1000 and len(data.text) == 5000
            and settings['annotation'] == relative(ANNOTATION))
    evaluation.config = config
    seed_everything(settings['seed'])
    model = (load_clip(target, BERT, device) if target.startswith('clip_')
             else load_fusion(target, BERT, device, config))
    tok = tokenizer(BERT)
    transform = target_transform(target, config)
    image_features, image_embeddings = [], []
    evaluated_texts = [None] * len(data.text)
    seen = []
    for record in manifest['batches']:
        item = checked_batch(run_dir, record, source=source)
        if item['image_ids'] != list(range(len(seen), len(seen) + len(item['image_ids']))):
            raise ValueError('Saved image batches do not follow dataset order')
        ids, texts = checked_text_batch(item, record, data)
        for text_id, text in zip(ids, texts):
            if evaluated_texts[text_id] is not None:
                raise ValueError('Duplicate adversarial caption ID')
            evaluated_texts[text_id] = text
        x = item['images']
        # White-box inputs already use the target resolution and preprocessing.
        if target != source:
            x = torch.stack([transform(image) for image in x])
        output = model.inference_image(NORMALIZE(x.to(device)))
        image_features.append(output['image_feat'].cpu().float())
        if not target.startswith('clip_'):
            image_embeddings.append(output['image_embed'].cpu().float())
        seen += item['image_ids']
        print(f'{target}: encoded {len(seen)}/{len(data)} images', flush=True)
    if seen != list(range(len(data))) or any(t is None for t in evaluated_texts):
        raise ValueError('Incomplete image or caption coverage')
    text_features, text_embeddings, text_atts = [], [], []
    text_batch_size = args.text_batch_size or settings['batch_size'] * 5
    for start in range(0, len(data.text), text_batch_size):
        inputs = tok(evaluated_texts[start:start + text_batch_size], padding='max_length',
                     truncation=True, max_length=30, return_tensors='pt').to(device)
        output = model.inference_text(inputs)
        text_features.append(output['text_feat'].cpu().float())
        if not target.startswith('clip_'):
            text_embeddings.append(output['text_embed'].cpu().float())
            text_atts.append(inputs.attention_mask.cpu())
    image_features, text_features = torch.cat(image_features), torch.cat(text_features)
    if target.startswith('clip_'):
        scores_i2t = (image_features @ text_features.T).numpy()
        scores_t2i = scores_i2t.T
        model_name = 'CLIP'
        encoder = 'ViT-B/16' if target == 'clip_vit' else 'RN101'
    else:
        wrapped = torch.nn.DataParallel(model, device_ids=[0] if device.type == 'cuda' else [])
        if not full:
            config['k_test'] = min(config['k_test'], len(data), len(data.text))
        scores_i2t, scores_t2i = evaluation.retrieval_score(
            wrapped, image_features, torch.cat(image_embeddings), text_features,
            torch.cat(text_embeddings), torch.cat(text_atts), len(data), len(data.text), device=device)
        scores_i2t, scores_t2i = scores_i2t.cpu().numpy(), scores_t2i.cpu().numpy()
        del wrapped
        model_name, encoder = target.upper(), 'ViT-B/16'
    result = dict(target=target, source_model=source, seed=settings['seed'], status='complete',
                  method='RAO-mixed11', white_box=target == source, attack_mode='joint',
                  image_source='saved_adversarial', text_source='saved_adversarial',
                  full_flickr30k=full, num_images=len(data), num_texts=len(data.text),
                  text_batch_size=text_batch_size,
                  scores_shape=[list(scores_i2t.shape), list(scores_t2i.shape)],
                  tf32_mode=args.tf32)
    if full:
        result['metrics'] = evaluation.itm_eval(
            scores_i2t, scores_t2i, data.img2txt, data.txt2img, model_name, encoder)
        result['tr1_asr'] = float(result['metrics']['txt_r1_ASR (txt_r1)'].split('(')[0])
        result['ir1_asr'] = float(result['metrics']['img_r1_ASR (img_r1)'].split('(')[0])
    else:
        result['note'] = 'ASR is computed only on the full test split.'
    out = run_dir / 'evaluations'
    out.mkdir(exist_ok=True)
    write_json(out / f'{target}.json', result)
    print(json.dumps(result, indent=2), flush=True)
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--targets', choices=TARGETS, nargs='+', default=list(TARGETS))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--tf32', choices=('on', 'off'), default='on')
    parser.add_argument('--text-batch-size', type=int)
    args = parser.parse_args()
    if args.text_batch_size is not None and args.text_batch_size < 1:
        parser.error('--text-batch-size must be positive')
    run_dir = resolve(args.run_dir)
    manifest = read_manifest(run_dir)
    device = device_for(args.gpu, args.device, args.threads)
    configure_tf32(args.tf32)
    for target in args.targets:
        evaluate_target(run_dir, manifest, target, args, device)


if __name__ == '__main__':
    main()

"""Model loading, preprocessing and checkpoint I/O."""
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from paths import (
    ROOT, ANNOTATION, DATA, BERT, EVAL_INDICES, CLIP_WEIGHTS, SOURCE_WEIGHTS,
    resolve, relative,
)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

TARGETS = ('clip_vit', 'clip_rn101', 'albef', 'tcl')
SOURCES = {
    'clip_vit': ('CLIP ViT-B/16', 224, 77),
    'clip_rn101': ('CLIP RN101', 224, 77),
    'albef': ('ALBEF', 384, 30),
    'tcl': ('TCL', 384, 30),
}
NORMALIZE = transforms.Normalize(
    (.48145466, .4578275, .40821073),
    (.26862954, .26130258, .27577711),
)


def configuration():
    config = json.loads((ROOT / 'configs/retrieval.json').read_text())
    config['bert_config'] = str(resolve(config['bert_config']))
    return config


def attack_configuration():
    return json.loads((ROOT / 'configs/attack.json').read_text())


def device_for(gpu='0', device='cuda', threads=2):
    gpu = str(gpu)
    if not gpu.isdecimal():
        raise ValueError('Each source experiment uses one physical GPU index')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu if device == 'cuda' else ''
    torch.set_num_threads(threads)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        torch.cuda.set_device(0)
    return torch.device(device)


def load_text_attacker(bert_dir, tok, device, max_length=77):
    from transformers import BertForMaskedLM
    from RAttacker import TextAttacker

    ref_model = BertForMaskedLM.from_pretrained(str(bert_dir), local_files_only=True)
    ref_model = ref_model.to(device).float().eval().requires_grad_(False)
    wrapped = torch.nn.DataParallel(
        ref_model, device_ids=[0] if device.type == 'cuda' else [],
    )
    wrapped.eval()
    return TextAttacker(
        wrapped, tok, cls=False, max_length=max_length,
        number_perturbation=1, topk=10, threshold_pred_score=.3,
    )


def attack_encoders(model, device):
    return ImageEncoder(model), TextEncoder(model)


def load_fusion(target, bert_dir, device, config):
    from models.model_retrieval import ALBEF
    from models.vit import interpolate_pos_embed

    tok = tokenizer(bert_dir)
    model = ALBEF(config=config, text_encoder=str(bert_dir), tokenizer=tok)
    checkpoint = torch.load(SOURCE_WEIGHTS[target], map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint)
    if target == 'tcl':
        for prefix in ('visual_encoder', 'visual_encoder_m'):
            key = prefix + '.pos_embed'
            state[key] = interpolate_pos_embed(state[key], getattr(model, prefix))
    for key in list(state):
        if 'bert' in key:
            state[key.replace('bert.', '')] = state.pop(key)
    message = model.load_state_dict(state, strict=False)
    required = ('visual_encoder.', 'vision_proj.', 'text_encoder.', 'text_proj.', 'itm_head.')
    missing = [key for key in message.missing_keys if key.startswith(required)]
    if missing:
        raise RuntimeError(f'Missing inference weights: {missing}')
    print(f'{target}: all inference weights loaded', flush=True)
    del checkpoint, state
    return model.to(device).float().eval().requires_grad_(False)


def read_manifest(run_dir):
    result = json.loads((Path(run_dir) / 'manifest.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('Adversarial generation is incomplete')
    return result


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def save_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def configure_tf32(mode):
    """Set matmul TF32; cuDNN TF32 remains enabled."""
    if mode not in ('on', 'off'):
        raise ValueError(f'Unknown TF32 mode: {mode}')
    torch.backends.cuda.matmul.allow_tf32 = mode == 'on'
    torch.backends.cudnn.allow_tf32 = True
    result = {
        'tf32_mode': mode,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
    }
    print('TF32: ' + json.dumps(result), flush=True)
    return result


def rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def source_transform(source='clip_vit'):
    resolution = SOURCES[source][1]
    if source in ('albef', 'tcl'):
        return transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=Image.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])


def dataset(annotation, image_root, max_images=None, source='clip_vit'):
    from dataset import paired_dataset

    result = paired_dataset(str(annotation), source_transform(source), str(image_root))
    if max_images is not None:
        if not 1 <= max_images <= len(result.ann):
            raise ValueError('--max-images is outside the annotation range')
        result.ann = result.ann[:max_images]
        result.image = result.image[:max_images]
        result.img2txt = {i: result.img2txt[i] for i in range(max_images)}
        n = sum(len(captions) for captions in result.img2txt.values())
        result.text = result.text[:n]
        result.txt2img = {i: result.txt2img[i] for i in range(n)}
    return result


def tokenizer(bert_dir):
    from models.tokenization_bert import BertTokenizer
    return BertTokenizer.from_pretrained(str(bert_dir), local_files_only=True)


def load_clip(target, bert_dir, device):
    from models.clip import clip

    filename = 'ViT-B-16.pt' if target == 'clip_vit' else 'RN101.pt'
    weight = CLIP_WEIGHTS / filename
    if not weight.is_file():
        raise FileNotFoundError(weight)
    model, _ = clip.load(str(weight), device=device, jit=False)
    model.set_tokenizer(tokenizer(bert_dir))
    return model.float().eval().requires_grad_(False)


def checked_text_batch(item, record, data):
    """Check saved captions against their dataset IDs."""
    ids, texts = item.get('text_ids'), item.get('adv_texts')
    expected = [text_id for image_id in item['image_ids'] for text_id in data.img2txt[image_id]]
    if ids != expected or ('text_ids' in record and record['text_ids'] != ids):
        raise ValueError('Adversarial text IDs disagree with the dataset or manifest')
    if not isinstance(texts, list) or len(texts) != len(expected):
        raise ValueError('Missing or invalid saved adversarial texts')
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('Missing or invalid saved adversarial texts')
    return ids, texts


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.inference_image(images)


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        inputs = SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)
        return self.model.inference_text(inputs)


def target_transform(target, config):
    if target in ('albef', 'tcl'):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
    ])


def checked_batch(run_dir, record, expected_ids=None, source='clip_vit'):
    path = Path(run_dir) / record['file']
    item = torch.load(path, map_location='cpu', weights_only=False)
    if item['image_ids'] != record['image_ids']:
        raise ValueError(f'Image IDs disagree with manifest: {path}')
    if expected_ids is not None and item['image_ids'] != expected_ids:
        raise ValueError('Image ordering changed')
    x = item['images']
    resolution = SOURCES[source][1]
    shape = (len(item['image_ids']), 3, resolution, resolution)
    if (x.dtype != torch.float32 or x.shape != shape or not torch.isfinite(x).all()
            or x.min() < 0 or x.max() > 1):
        raise ValueError(f'Invalid image tensor: {path}')
    return item

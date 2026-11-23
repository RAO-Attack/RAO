"""Project paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/flickr30k'
ANNOTATION = DATA / 'flickr30k_test.json'
EVAL_INDICES = ROOT / 'data/eval_indices'
BERT = ROOT / 'weights/bert-base-uncased'
CLIP_WEIGHTS = ROOT / 'weights/clip'
SOURCE_WEIGHTS = {
    'clip_vit': CLIP_WEIGHTS / 'ViT-B-16.pt',
    'clip_rn101': CLIP_WEIGHTS / 'RN101.pt',
    'albef': ROOT / 'weights/albef_flickr.pth',
    'tcl': ROOT / 'weights/tcl_flickr.pth',
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT))

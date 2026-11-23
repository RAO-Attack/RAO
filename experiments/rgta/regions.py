import math
import torch
import torch.nn.functional as F

def validate_weights(global_weight, local_weight):
    if not all((math.isfinite(v) and v >= 0 for v in (global_weight, local_weight))) or not math.isfinite(global_weight + local_weight) or global_weight + local_weight <= 0:
        raise ValueError('RGTA global/local weights must be finite, nonnegative and not both zero')

def crop_boxes(height, width):
    if min(height, width) < 1:
        raise ValueError('RGTA images must have nonempty spatial dimensions')
    crop_h = max(1, min(height, int(round(0.5 * height))))
    crop_w = max(1, min(width, int(round(0.5 * width))))
    bottom, right = (height - crop_h, width - crop_w)
    return [(0, 0, crop_h, crop_w), (0, right, crop_h, crop_w), (bottom, 0, crop_h, crop_w), (bottom, right, crop_h, crop_w), (bottom // 2, right // 2, crop_h, crop_w)]

@torch.no_grad()
def local_image_features(encoder, normalization, images):
    """Return five normalized crop features per image, shaped [B, 5, D]."""
    if images.ndim != 4 or not images.shape[0] or (not images.is_floating_point()):
        raise ValueError('RGTA requires nonempty floating-point BCHW images')
    height, width = images.shape[-2:]
    result = []
    for top, left, crop_h, crop_w in crop_boxes(height, width):
        crop = images.detach()[:, :, top:top + crop_h, left:left + crop_w]
        crop = F.interpolate(crop, size=(height, width), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        pixels = normalization(crop) if normalization is not None else crop
        features = encoder(pixels)['image_feat'].detach()
        if features.ndim != 2 or features.shape[0] != images.shape[0] or (not torch.isfinite(features).all()):
            raise ValueError('RGTA image encoder must return finite [B,D] image features')
        result.append(features)
    return torch.stack(result, dim=1)


"""Mixed11 image attack and fixed-five-region G1/L1 text attack."""
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms


def rgta_settings():
    return {'global_weight': 1.0, 'local_weight': 1.0}


def hqa_resource_paths():
    root = Path(__file__).resolve().parent / 'data/text'
    return {
        'counter_fitted': root / 'counter-fitted-vectors.txt',
        'word_similarity': root / 'mat_sim_0.4.txt',
    }


class Attacker:

    def __init__(self, model_inf, model_infi, model_inft, img_attacker, txt_attacker):
        self.model_inf = model_inf
        self.model_infi = model_infi
        self.model_inft = model_inft
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    def attack91(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
        """Fixed-five-region G1/L1 text search followed by historical image search."""
        start_time = time.time()
        with torch.no_grad():
            origin_img_output = self.model_infi(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]
        from experiments.rgta.regions import local_image_features
        options = rgta_settings()
        local_supervisions = None
        if options['local_weight'] > 0:
            local_features = local_image_features(self.model_infi, self.img_attacker.normalization, imgs)
            local_supervisions = local_features[txt2img]
        adv_txts = self.txt_attacker.img_guided_attack_rgta(self.model_inft, txts, txt2img, img_embeds=img_supervisions, local_img_embeds=local_supervisions, **options)
        if len(adv_txts) != len(txts):
            raise ValueError('Text attack must return one adversarial caption per input caption')
        end_time = time.time()
        execuate_time = end_time - start_time
        start_time1 = time.time()
        with torch.no_grad():
            adv_txt_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            adv_txt_ids = adv_txt_input.input_ids.to(device)
            adv_txt_mask = adv_txt_input.attention_mask.to(device)
            adv_txt_embeds = self.model_inft(adv_txt_ids, adv_txt_mask)['text_feat'].detach()
            ori_txt_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            ori_txt_ids = ori_txt_input.input_ids.to(device)
            ori_txt_mask = ori_txt_input.attention_mask.to(device)
            ori_txt_embeds = self.model_inft(ori_txt_ids, ori_txt_mask)['text_feat'].detach()
        adv_imgs, last_adv_imgs = self.img_attacker.txt_guided_attack11(self.model_infi, imgs, txt2img, device, scales=scales, txt_embeds=adv_txt_embeds, ori_txt_embeds=ori_txt_embeds)
        end_time1 = time.time()
        execuate_time1 = end_time1 - start_time1
        print(execuate_time1)
        return (adv_imgs, adv_txts)

class ImageAttacker:

    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255, sample_numbers=5):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.sample_numbers = sample_numbers

    def image_attack_pixels(self, images):
        return images

    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img, ori_txt_embeds=None):
        """Matched clean/adv captions: -10; unmatched adv captions: +1."""
        sim = adv_imgs_embeds @ txts_embeds.T
        caption_idx = torch.arange(sim.shape[1], device=sim.device)
        matched_img_idx = torch.as_tensor(txt2img, device=sim.device, dtype=torch.long)
        if matched_img_idx.numel() != sim.shape[1]:
            raise ValueError('txt2img length must match number of captions')
        weights = torch.ones_like(sim)
        weights[matched_img_idx, caption_idx] = -10.0
        loss = (sim * weights).sum(dim=-1).mean()
        if ori_txt_embeds is not None:
            if ori_txt_embeds.shape != txts_embeds.shape:
                raise ValueError('Clean and adversarial text features must have identical shape')
            sim_clean = adv_imgs_embeds @ ori_txt_embeds.T
            loss = loss - 10.0 * sim_clean[matched_img_idx, caption_idx].sum() / sim_clean.shape[0]
        return loss

    def get_crop_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs
        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        local_transform = transforms.RandomResizedCrop(ori_shape, scale=(0.5, 0.8))
        result = []
        for i in range(scales):
            reversed_imgs = local_transform(imgs)
            result.append(reversed_imgs)
        return torch.cat([imgs] + result, 0)

    def configure_mixed_augmentation(self, noise_sigma=0.01, translation_fraction=0.15, resize_min=0.75, translation_noise_sigma=0.0025, extra_repeats=2, noise_sigmas=(0.005, 0.01), resize_restore_scales=(0.5, 0.75)):
        """Configure the mixed11 recipe from configs/attack.json."""
        if not 0.0 <= noise_sigma < float('inf'):
            raise ValueError('Mixed augmentation noise sigma must be finite and nonnegative')
        if not 0.0 <= translation_fraction < 0.5:
            raise ValueError('Mixed translation fraction must be finite and in [0,0.5)')
        if not 0.75 <= resize_min <= 1.0:
            raise ValueError('Mixed resize minimum must be finite and in [0.75,1]')
        if not 0.0 <= translation_noise_sigma < float('inf'):
            raise ValueError('Mixed translation noise sigma must be finite and nonnegative')
        if type(extra_repeats) is not int or extra_repeats not in (1, 2):
            raise ValueError('Mixed noise/translation/resize repeat count must be 1 or 2')
        if noise_sigmas is not None:
            noise_sigmas = tuple(noise_sigmas)
            if len(noise_sigmas) != extra_repeats or any((not 0.0 <= sigma < float('inf') for sigma in noise_sigmas)):
                raise ValueError('Supply one finite nonnegative noise sigma per noise view')
        resize_restore_scales = tuple(resize_restore_scales)
        if any((not 0.0 < ratio <= 1.0 for ratio in resize_restore_scales)):
            raise ValueError('Resize/restore scales must be finite and in (0,1]')
        self.crop_noise_sigma = noise_sigma
        self.mixed_translation_fraction = translation_fraction
        self.mixed_resize_min = resize_min
        self.mixed_translation_noise_sigma = translation_noise_sigma
        self.mixed_extra_repeats = extra_repeats
        self.mixed_noise_sigmas = noise_sigmas
        self.mixed_resize_restore_scales = resize_restore_scales
        self.augmentation_view_count = 2 + 3 * extra_repeats + len(resize_restore_scales)
        self.checkpoint_augmented_views = extra_repeats == 2
        self.get_crop_imgs = self.get_mixed_imgs

    def augmentation_views(self):
        return self.augmentation_view_count


    def get_mixed_imgs(self, imgs, scales=None, device='cuda'):
        """Whole + two crops + independently sampled noise/translation/resize views."""
        if scales is None:
            return imgs
        if scales != self.augmentation_view_count:
            raise ValueError(f'Mixed augmentation needs {self.augmentation_view_count} additional views')
        if imgs.ndim != 4 or min(imgs.shape[-2:]) < 2:
            raise ValueError('Mixed augmentation requires BCHW images with height and width >= 2')
        height, width = imgs.shape[-2:]
        crops = ImageAttacker.get_crop_imgs(self, imgs, 2, device)
        result = [crops]
        noise_sigmas = self.mixed_noise_sigmas or (self.crop_noise_sigma,) * self.mixed_extra_repeats
        for sigma in noise_sigmas:
            noisy = (imgs + sigma * torch.randn_like(imgs)).clamp(0, 1)
            result.append(noisy)
        max_y = int(round(self.mixed_translation_fraction * height))
        max_x = int(round(self.mixed_translation_fraction * width))
        padded = torch.nn.functional.pad(imgs, (max_x, max_x, max_y, max_y), mode='reflect')
        for _ in range(self.mixed_extra_repeats):
            dy = int(torch.randint(-max_y, max_y + 1, ()).item())
            dx = int(torch.randint(-max_x, max_x + 1, ()).item())
            top, left = (max_y - dy, max_x - dx)
            translated = padded[:, :, top:top + height, left:left + width]
            if self.mixed_translation_noise_sigma > 0:
                translated = (translated + self.mixed_translation_noise_sigma * torch.randn_like(translated)).clamp(0, 1)
            result.append(translated)
        for _ in range(self.mixed_extra_repeats):
            ratio = float(torch.empty(()).uniform_(self.mixed_resize_min, 1.0).item())
            small_h, small_w = (max(1, int(round(height * ratio))), max(1, int(round(width * ratio))))
            small = torch.nn.functional.interpolate(imgs, size=(small_h, small_w), mode='bilinear', align_corners=False, antialias=True)
            gap_h, gap_w = (height - small_h, width - small_w)
            pad_top = int(torch.randint(gap_h + 1, ()).item())
            pad_left = int(torch.randint(gap_w + 1, ()).item())
            resized = torch.nn.functional.pad(small, (pad_left, gap_w - pad_left, pad_top, gap_h - pad_top), mode='reflect')
            result.append(resized)
        for ratio in self.mixed_resize_restore_scales:
            small_shape = (max(1, int(height * ratio)), max(1, int(width * ratio)))
            small = transforms.Resize(small_shape, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(imgs).clamp(0.0, 1.0)
            restored = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(small)
            result.append(restored)
        return torch.cat(result, dim=0)

    def txt_guided_attack11(self, model, imgs, txt2img, device, scales=None, txt_embeds=None, ori_txt_embeds=None):
        from experiments.mixed_views.memory import CheckpointedViewEncoder
        model = CheckpointedViewEncoder(model, imgs.shape[0])
        model.eval()
        b, _, w, h = imgs.shape
        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) + 1
        delta = torch.zeros_like(imgs)
        delta = delta.to(device)
        orig_delta = delta.clone().detach()
        last_adv_imgs = None
        last_adv_delta = None
        last_adv_grad = torch.zeros_like(imgs)
        last_adv_grad = last_adv_grad.to(device)
        start_time = time.time()
        ratio_list = []
        imgs_output = model(self.normalization(self.image_attack_pixels(imgs)))
        imgs_embeds = imgs_output['image_feat']
        if txt_embeds is None:
            raise ValueError('txt_guided_attack11 requires txt_embeds')
        txt2img_index = torch.as_tensor(txt2img, device=imgs.device, dtype=torch.long)

        def io_loss(image_features, model_output=None, subset=None):
            loss = self.loss_func(image_features, txt_embeds, txt2img_index, ori_txt_embeds=ori_txt_embeds)
            return loss
        loss_bar = io_loss(imgs_embeds, imgs_output)
        all_grads = []
        all_grads.append(delta)
        momentum = torch.zeros_like(imgs).detach().to(device)
        local_transform = transforms.RandomResizedCrop(w, scale=(0.5, 0.8))
        se_num = self.augmentation_views()
        scales = [0.5, 0.75, 1.25, 1.5]
        exp_num = 5
        for step in range(self.steps):
            delta = delta.clone().detach().requires_grad_(True)
            adv_imgs = imgs.detach() + delta
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
            if last_adv_delta != None:
                samples = []
                samples1 = []
                clone_adv_delta = delta.clone()
                clone_adv_grad = grad.clone()
                loss_list = []
                stop_flag = False
                max_loss = float('-inf')
                max_loss0 = float('-inf')
                max_loss1 = float('-inf')
                max_loss2 = float('-inf')
                grad1 = torch.zeros_like(imgs)
                grad1 = grad1.to(device)
                new_grad = torch.zeros_like(imgs)
                new_grad = new_grad.to(device)
                for k in range(exp_num):
                    coeffs_pos = torch.rand(step + 1, device=device)
                    coeffs_pos = coeffs_pos / coeffs_pos.sum()
                    new_grad0 = sum((coeffs_pos[t] * all_grads[t] for t in range(step)))
                    coeffs_neg = torch.randn(step + 1, device=device)
                    while (coeffs_neg > 0).all() or (coeffs_neg < 0).all() or coeffs_neg.sum().abs() < 1e-06:
                        coeffs_neg = torch.randn(step + 1, device=device)
                    coeffs_neg /= coeffs_neg.sum()
                    new_grad1 = sum((coeffs_neg[t] * all_grads[t] for t in range(step)))
                    new_grad = new_grad + new_grad1 + new_grad0
                    delta0 = last_adv_delta + self.step_size * new_grad0.sign()
                    delta1 = last_adv_delta + self.step_size * new_grad1.sign()
                    delta0 = torch.clamp(delta0, -self.eps, self.eps)
                    delta1 = torch.clamp(delta1, -self.eps, self.eps)
                    adv_imgs0 = imgs.detach() + delta0
                    adv_imgs1 = imgs.detach() + delta1
                    adv_imgs0 = torch.clamp(adv_imgs0, 0.0, 1.0)
                    adv_imgs1 = torch.clamp(adv_imgs1, 0.0, 1.0)
                    if self.normalization is not None:
                        adv_imgs_output0 = model(self.normalization(self.image_attack_pixels(adv_imgs0)))
                        adv_imgs_output1 = model(self.normalization(self.image_attack_pixels(adv_imgs1)))
                    else:
                        adv_imgs_output = model(self.image_attack_pixels(adv_imgs))
                    adv_imgs_embeds0 = adv_imgs_output0['image_feat']
                    adv_imgs_embeds1 = adv_imgs_output1['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        loss0 = io_loss(adv_imgs_embeds0, adv_imgs_output0)
                        loss1 = io_loss(adv_imgs_embeds1, adv_imgs_output1)
                    if loss0.item() > loss_bar:
                        max_loss0 = loss0.item()
                        best_adv_imgs0 = adv_imgs0.detach()
                        best_delta0 = delta0.detach()
                        best_grad0 = new_grad0.detach()
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        delta_ = best_delta0.clone().detach().requires_grad_(True)
                        adv_imgs0 = torch.clamp(imgs + delta_, 0.0, 1.0)
                        scaled_imgs = self.get_crop_imgs(self.image_attack_pixels(adv_imgs0), se_num, device)
                        if self.normalization is not None:
                            adv_imgs_output = model(self.normalization(scaled_imgs))
                        else:
                            adv_imgs_output = model(scaled_imgs)
                        adv_imgs_embeds = adv_imgs_output['image_feat']
                        model.zero_grad()
                        with torch.enable_grad():
                            loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                            for i in range(se_num + 1):
                                loss_item = io_loss(adv_imgs_embeds[i * b:i * b + b], adv_imgs_output, slice(i * b, (i + 1) * b))
                                loss += loss_item
                        grad = torch.autograd.grad(loss, [delta_])[0]
                        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                        grad1 = grad + grad1
                    if loss1.item() > loss_bar:
                        max_loss1 = loss1.item()
                        best_adv_imgs1 = adv_imgs1.detach()
                        best_delta1 = delta1.detach()
                        best_grad1 = new_grad1.detach()
                        delta_ = best_delta1.clone().detach().requires_grad_(True)
                        adv_imgs1 = torch.clamp(imgs + delta_, 0.0, 1.0)
                        scaled_imgs = self.get_crop_imgs(self.image_attack_pixels(adv_imgs1), se_num, device)
                        if self.normalization is not None:
                            adv_imgs_output = model(self.normalization(scaled_imgs))
                        else:
                            adv_imgs_output = model(scaled_imgs)
                        adv_imgs_embeds = adv_imgs_output['image_feat']
                        model.zero_grad()
                        with torch.enable_grad():
                            loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                            for i in range(se_num + 1):
                                loss_item = io_loss(adv_imgs_embeds[i * b:i * b + b], adv_imgs_output, slice(i * b, (i + 1) * b))
                                loss += loss_item
                        grad = torch.autograd.grad(loss, [delta_])[0]
                        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                        grad1 = grad + grad1
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for adv in [adv_imgs]:
                    delta_ = torch.zeros_like(imgs)
                    delta_ = delta_.to(device)
                    delta_ = delta_.clone().detach().requires_grad_(True)
                    adv_imgs = adv_imgs.detach() + delta_
                    adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
                    scaled_imgs = self.get_crop_imgs(self.image_attack_pixels(adv_imgs), se_num, device)
                    if self.normalization is not None:
                        adv_imgs_output = model(self.normalization(scaled_imgs))
                    else:
                        adv_imgs_output = model(scaled_imgs)
                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        for i in range(se_num + 1):
                            loss_item = io_loss(adv_imgs_embeds[i * b:i * b + b], adv_imgs_output, slice(i * b, (i + 1) * b))
                            loss += loss_item
                    grad = torch.autograd.grad(loss, [delta_])[0]
                    grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                    grad1 = grad + grad1
                grad = grad1
                new_grad = 0.5 * new_grad / (step * 2) + clone_adv_grad
                new_delta = last_adv_delta + self.step_size * new_grad.sign()
                new_delta = torch.clamp(delta, -self.eps, self.eps)
                with torch.no_grad():
                    delta = clone_adv_delta + self.step_size * grad.sign()
                    delta = torch.clamp(delta, -self.eps, self.eps)
                last_adv_delta = clone_adv_delta.clone()
                last_adv_grad = clone_adv_grad.clone()
                all_grads.append(grad)
                adv_imgs = imgs.detach() + delta
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
                momentum = grad
            else:
                last_adv_delta = delta.clone()
                scaled_imgs = self.get_crop_imgs(self.image_attack_pixels(adv_imgs), se_num, device)
                if self.normalization is not None:
                    adv_imgs_output = model(self.normalization(scaled_imgs))
                    adv_imgs_output1 = model(self.normalization(self.image_attack_pixels(adv_imgs)))
                else:
                    adv_imgs_output = model(scaled_imgs)
                adv_imgs_embeds = adv_imgs_output['image_feat']
                adv_imgs_embeds1 = adv_imgs_output1['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                    for i in range(se_num + 1):
                        loss_item = io_loss(adv_imgs_embeds[i * b:i * b + b], adv_imgs_output, slice(i * b, (i + 1) * b))
                        loss += loss_item
                grad = torch.autograd.grad(loss, [delta])[0]
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                momentum = grad
                with torch.no_grad():
                    delta = delta + self.step_size * grad.sign()
                    delta = torch.clamp(delta, -self.eps, self.eps)
                adv_imgs = imgs.detach() + delta
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
                all_grads.append(grad)
                best_adv_imgs1 = adv_imgs
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'The function execution time: {elapsed_time} seconds')
        adv_imgs = imgs.detach() + delta
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        return (adv_imgs, last_adv_imgs)

class TextAttacker:

    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10, threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2]):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios

    def _load_hqa_resources(self):
        """Load HQA's vocabulary and pickled neighbours once, only on demand."""
        if hasattr(self, '_hqa_word2idx'):
            return
        import pickle
        paths = hqa_resource_paths()
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f'HQA text resource not found: {path}')
        idx2word, word2idx = ({}, {})
        with paths['counter_fitted'].open(encoding='utf-8') as stream:
            for line in stream:
                word = line.split()[0]
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1
        with paths['word_similarity'].open('rb') as stream:
            similarities = pickle.load(stream)
        if len(similarities) != len(idx2word):
            raise ValueError('HQA vocabulary and similarity-list lengths disagree')
        self._hqa_idx2word = idx2word
        self._hqa_word2idx = word2idx
        self._hqa_similarities = similarities
        print(f'HQA text resources loaded: {len(idx2word)} words', flush=True)

    @torch.no_grad()
    def _encode_hqa_candidates(self, net, texts, device):
        embeddings = []
        for start in range(0, len(texts), 64):
            inputs = self.tokenizer(texts[start:start + 64], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
            features = net(inputs.input_ids, inputs.attention_mask)['text_feat']
            embeddings.append(features[:, 0, :] if self.cls else features.flatten(1))
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def img_guided_attack_rgta(self, net, texts, txt2img, img_embeds=None, *, local_img_embeds=None, global_weight=1.0, local_weight=1.0):
        """Score one-word substitutions with global and mean local image features."""
        from experiments.rgta.regions import validate_weights
        validate_weights(global_weight, local_weight)
        if img_embeds is None or img_embeds.ndim != 2 or img_embeds.shape[0] != len(texts) or (not torch.isfinite(img_embeds).all()):
            raise ValueError('RGTA requires finite matched image features [num_texts,D]')
        if len(txt2img) != len(texts):
            raise ValueError('txt2img length must match the number of captions')
        anchor = img_embeds
        if local_weight > 0 and texts:
            if local_img_embeds is None or tuple(local_img_embeds.shape) != (len(texts), 5, img_embeds.shape[1]) or local_img_embeds.device != img_embeds.device or (not torch.isfinite(local_img_embeds).all()):
                raise ValueError('RGTA requires finite matched local features [num_texts,5,D] on the same device')
            total = global_weight + local_weight
            anchor = global_weight / total * img_embeds + local_weight / total * local_img_embeds.mean(dim=1)
        return self.img_guided_attack_hqa(net, texts, txt2img, img_embeds=anchor)

    @torch.no_grad()
    def img_guided_attack_hqa(self, net, texts, txt2img, img_embeds=None):
        """Use CF neighbours above 0.4, with BERT fallback for out-of-vocabulary words."""
        if img_embeds is None or img_embeds.shape[0] != len(texts):
            raise ValueError('HQA requires one matched image-feature row per caption')
        if len(txt2img) != len(texts):
            raise ValueError('txt2img length must match the number of captions')
        if not texts:
            return []
        if self.num_perturbation < 1:
            return list(texts)
        self._load_hqa_resources()
        device = img_embeds.device
        ref_model = getattr(self.ref_net, 'module', self.ref_net)
        inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt').to(device)
        logits = ref_model(inputs.input_ids, attention_mask=inputs.attention_mask).logits
        scores, predictions = torch.topk(logits, self.topk, -1)
        features = net(inputs.input_ids, inputs.attention_mask)['text_feat']
        origin_embeds = features[:, 0, :] if self.cls else features.flatten(1)
        final_adverse = []
        for i, text in enumerate(texts):
            words, _, keys = self._tokenize(text)
            importance = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)
            positions = sorted(enumerate(importance), key=lambda item: item[1], reverse=True)
            final_words = copy.deepcopy(words)
            best_score = -10
            best_edit = None
            for position, _ in positions:
                word = words[position]
                if word in filter_words or keys[position][0] > self.max_length - 2:
                    continue
                if word in self._hqa_word2idx:
                    row = self._hqa_similarities[self._hqa_word2idx[word]]
                    substitutes = [self._hqa_idx2word[index] for similarity, index in row if similarity > 0.4]
                else:
                    begin, end = keys[position]
                    substitutes = get_substitues(predictions[i, begin:end], self.tokenizer, ref_model, 1, scores[i, begin:end], self.threshold_pred_score)
                replacements = [' '.join(final_words)]
                available = [word]
                for substitute in substitutes:
                    if substitute == word or '##' in substitute or substitute in filter_words:
                        continue
                    candidate_words = copy.deepcopy(final_words)
                    candidate_words[position] = substitute
                    available.append(substitute)
                    replacements.append(' '.join(candidate_words))
                candidate_features = self._encode_hqa_candidates(net, replacements, device)
                loss = self.loss_func(candidate_features, img_embeds, i)
                candidate = loss.argmax()
                if loss[candidate] > best_score:
                    best_score = loss[candidate]
                    best_edit = (position, available[candidate])
            if best_edit is not None:
                position, replacement = best_edit
                final_words[position] = replacement
            final_adverse.append(' '.join(final_words))
        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def _tokenize(self, text):
        words = text.split(' ')
        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)
        return (words, sub_words, keys)

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device
        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]
        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            masked_text_input.input_ids = masked_text_input.input_ids.to(device)
            masked_text_input.attention_mask = masked_text_input.attention_mask.to(device)
            masked_output = net(masked_text_input.input_ids, masked_text_input.attention_mask)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)
        criterion = torch.nn.KLDivLoss(reduction='none')
        import_scores = criterion(masked_embeds.log_softmax(dim=-1), origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))
        return import_scores.sum(dim=-1)
filter_words = [
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all',
    'almost', 'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an',
    'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are',
    'aren', "aren't", 'around', 'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind',
    'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'but', 'by', 'can',
    'cannot', 'could', 'couldn', "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don',
    "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere', 'empty', 'enough', 'even',
    'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for', 'former',
    'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
    'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is',
    'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll',
    'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more', 'moreover', 'most',
    'mostly', 'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn', "needn't",
    'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not',
    'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or',
    'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please',
    's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't",
    'somehow', 'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll",
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter',
    'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'this', 'those', 'through',
    'throughout', 'thru', 'thus', 'to', 'too', 'toward', 'towards', 'under', 'unless', 'until',
    'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't", 'we', 'were', 'weren', "weren't",
    'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
    'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither',
    'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
    "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some',
    '"', ',', 'b', '&', '!', '@', '%', '^', '*', '(', ')', '-', '-', '+', '=', '<', '>', '|',
    ':', ';', '～', '·',
]
filter_words = set(filter_words)

def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    sub_len, k = substitutes.size()
    if sub_len == 0:
        return words
    elif sub_len == 1:
        for i, j in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    elif use_bpe == 1:
        words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
    else:
        return words
    return words

def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    try:
        device = mlm_model.module.device
    except:
        device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]
    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    all_substitutes = torch.tensor(all_substitutes)
    all_substitutes = all_substitutes[:24].to(device)
    N, L = all_substitutes.size()
    try:
        word_predictions = mlm_model(all_substitutes)[0]
    except:
        word_predictions = mlm_model.module(all_substitutes)[0]
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words

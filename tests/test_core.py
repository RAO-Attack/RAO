"""Tests for image views, regions, loss and retrieval metrics."""
import json
import unittest

import numpy as np
import torch

from common import (ANNOTATION, DATA, EVAL_INDICES, attack_configuration, dataset,
                    restore_rng, rng_state, seed_everything)
from experiments.rgta.regions import crop_boxes
from RAttacker import ImageAttacker
import evaluation


class RetrievalCoreTests(unittest.TestCase):
    def test_mixed11_preserves_whole_view_and_backpropagates(self):
        images = torch.linspace(.1, .9, 2 * 3 * 24 * 24).reshape(2, 3, 24, 24).requires_grad_()
        attacker = ImageAttacker(lambda x: x)
        attacker.configure_mixed_augmentation(**attack_configuration()['mixed11'])
        views = attacker.get_mixed_imgs(images, 10, 'cpu')
        self.assertEqual(tuple(views.shape), (22, 3, 24, 24))
        torch.testing.assert_close(views[:2], images, rtol=0, atol=0)
        self.assertEqual(attacker.augmentation_views() + 1, 11)
        views.sum().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertTrue((images.grad.abs().sum((1, 2, 3)) > 0).all())

    def test_resume_restores_transform_randomness(self):
        attacker = ImageAttacker(lambda x: x)
        attacker.configure_mixed_augmentation(**attack_configuration()['mixed11'])
        images = torch.ones(2, 3, 24, 24) / 2
        seed_everything(42)
        state = rng_state()
        expected = attacker.get_mixed_imgs(images, 10, 'cpu')
        torch.rand(100)
        restore_rng(state)
        actual = attacker.get_mixed_imgs(images, 10, 'cpu')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fixed_five_regions_use_half_side_length(self):
        self.assertEqual(crop_boxes(224, 224), [(0, 0, 112, 112), (0, 112, 112, 112),
                         (112, 0, 112, 112), (112, 112, 112, 112), (56, 56, 112, 112)])

    def test_image_objective_keeps_both_positive_terms_and_adversarial_negatives(self):
        images = torch.tensor([[.3, .8], [.7, -.2]], requires_grad=True)
        adversarial = torch.tensor([[1., 0.], [.1, 1.], [-1., .2], [.5, -.5]])
        clean = adversarial * .6
        mapping = [0, 0, 1, 1]
        actual = ImageAttacker(None).loss_func(images, adversarial, mapping, clean)
        expected = sum(-10 * (images[i] @ adversarial[j]) - 10 * (images[i] @ clean[j])
                       if i == mapping[j] else images[i] @ adversarial[j]
                       for i in range(2) for j in range(4)) / 2
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, images, retain_graph=True)[0],
                                   torch.autograd.grad(expected, images)[0])

    def test_split_and_clean_correct_asr_denominators(self):
        data = dataset(ANNOTATION, DATA, source='clip_vit')
        self.assertEqual((len(data), len(data.text)), (1000, 5000))
        i2t = np.zeros((1000, 5000), dtype=np.float32)
        for i, captions in data.img2txt.items():
            i2t[i, captions] = 10
        t2i = i2t.T.copy()
        clean_tr = np.load(EVAL_INDICES / 'TCL_tr1_rank_index.npy')
        clean_ir = np.load(EVAL_INDICES / 'TCL_ir1_rank_index.npy')
        image_id, caption_id = int(clean_tr[0]), int(clean_ir[0])
        wrong_text = next(t for t in range(5000) if data.txt2img[t] != image_id)
        i2t[image_id, wrong_text] = 20
        t2i[caption_id, (data.txt2img[caption_id] + 1) % 1000] = 20
        result = evaluation.itm_eval(i2t, t2i, data.img2txt, data.txt2img, 'TCL', 'ViT-B/16')
        tr = float(result['txt_r1_ASR (txt_r1)'].split('(')[0])
        ir = float(result['img_r1_ASR (img_r1)'].split('(')[0])
        self.assertEqual(tr, round(100 / len(clean_tr), 2))
        self.assertEqual(ir, round(100 / len(clean_ir), 2))


if __name__ == '__main__':
    unittest.main()

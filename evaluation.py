"""Fusion reranking and attack success rates on clean-correct queries."""
import numpy as np
import torch
import utils
from types import SimpleNamespace
from paths import EVAL_INDICES
config = {}
args = SimpleNamespace(original_rank_index_path=str(EVAL_INDICES))

@torch.no_grad()
def retrieval_score(model, image_feats, image_embeds, text_feats, text_embeds, text_atts, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Evaluation Direction Similarity With Bert Attack:'
    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.module.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device), attention_mask=text_atts[topk_idx].to(device), encoder_hidden_states=encoder_output, encoder_attention_mask=encoder_att, return_dict=True, mode='fusion')
        score = model.module.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.module.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device), attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device), encoder_hidden_states=encoder_output, encoder_attention_mask=encoder_att, return_dict=True, mode='fusion')
        score = model.module.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score
    return (score_matrix_i2t, score_matrix_t2i)

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img, model_name, image_encoder):
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        rank = 1e+20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]
    original_rank_index_path = args.original_rank_index_path
    if model_name == 'CLIP':
        if image_encoder in ['RN50', 'RN101']:
            origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_CNN_tr1_rank_index.npy')
            origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_CNN_tr5_rank_index.npy')
            origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_CNN_tr10_rank_index.npy')
        else:
            origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_ViT_tr1_rank_index.npy')
            origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_ViT_tr5_rank_index.npy')
            origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_ViT_tr10_rank_index.npy')
    elif model_name == 'ALBEF' or model_name == 'TCL':
        origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_tr1_rank_index.npy')
        origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_tr5_rank_index.npy')
        origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_tr10_rank_index.npy')
    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2)
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]
    if model_name == 'CLIP':
        if image_encoder in ['RN50', 'RN101']:
            origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_CNN_ir1_rank_index.npy')
            origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_CNN_ir5_rank_index.npy')
            origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_CNN_ir10_rank_index.npy')
        else:
            origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_ViT_ir1_rank_index.npy')
            origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_ViT_ir5_rank_index.npy')
            origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_ViT_ir10_rank_index.npy')
    elif model_name == 'ALBEF' or model_name == 'TCL':
        origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_ir1_rank_index.npy')
        origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_ir5_rank_index.npy')
        origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_ir10_rank_index.npy')
    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2)
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2)
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)
    eval_result = {'txt_r1_ASR (txt_r1)': f'{asr_tr1}({tr1})', 'txt_r5_ASR (txt_r5)': f'{asr_tr5}({tr5})', 'txt_r10_ASR (txt_r10)': f'{asr_tr10}({tr10})', 'img_r1_ASR (img_r1)': f'{asr_ir1}({ir1})', 'img_r5_ASR (img_r5)': f'{asr_ir5}({ir5})', 'img_r10_ASR (img_r10)': f'{asr_ir10}({ir10})'}
    return eval_result

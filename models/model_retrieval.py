from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
import torch
from torch import nn
import torch.nn.functional as F

class ALBEF(nn.Module):

    def __init__(self, text_encoder=None, tokenizer=None, config=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.distill = config['distill']
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        self.visual_encoder = VisionTransformer(img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06))
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)
        self.visual_encoder_m = VisionTransformer(img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-06))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m], [self.vision_proj, self.vision_proj_m], [self.text_encoder, self.text_encoder_m], [self.text_proj, self.text_proj_m]]
        self.copy_params()
        self.register_buffer('image_queue', torch.randn(embed_dim, self.queue_size))
        self.register_buffer('text_queue', torch.randn(embed_dim, self.queue_size))
        self.register_buffer('idx_queue', torch.full((1, self.queue_size), -100))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def inference_text(self, text_input):
        text_output = self.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embed[:, 0, :]), dim=-1)
        return {'text_feat': text_feat, 'text_embed': text_embed}

    def inference_image(self, image):
        image_embed = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embed[:, 0, :]), dim=-1)
        return {'image_feat': image_feat, 'image_embed': image_embed}

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

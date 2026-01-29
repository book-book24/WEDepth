import torch
import torch.nn as nn
from sympy import expand
from torch.nn.modules.utils import _pair
from operator import mul
import math
from functools import reduce

from .transformer_dinov2 import Dinov2Model
from .merge import ExtractionPattern


class PromptDINOV2(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.patch_size = config["model"]["patch_size"]

        patch_size = _pair(self.patch_size)

        self.num_prompt = config["model"]["prompt_num"]
        self.prompt_dim = config["model"]["prompt_dim"]
        self.out_dim = config["model"]["out_dim"]
        self.token_dim = config["model"]["token_dim"]
        self.ext_num = config["model"]["ext_num"]
        self.model_type = config["model"]["dino_type"]
        self.layer_idx = config["model"]["layer_idx"]
        # self.hidden_state_layer = cfg.hidden_state_layer

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.prompt_dim))
        self.prompt_embeddings_1 = nn.Parameter(torch.zeros(1, self.num_prompt, self.prompt_dim))
        self.prompt_embeddings_2 = nn.Parameter(torch.zeros(1, self.num_prompt, self.prompt_dim))
        self.prompt_embeddings_3 = nn.Parameter(torch.zeros(1, self.num_prompt, self.prompt_dim))
        self.prompt_embeddings_4 = nn.Parameter(torch.zeros(1, self.num_prompt, self.prompt_dim))
        nn.init.uniform_(self.prompt_embeddings_1.data, -val, val)
        nn.init.uniform_(self.prompt_embeddings_2.data, -val, val)
        nn.init.uniform_(self.prompt_embeddings_3.data, -val, val)
        nn.init.uniform_(self.prompt_embeddings_4.data, -val, val)

        self.make_mask_all_dateset(config)
        self.model_hf = Dinov2Model.from_pretrained(self.model_type)

        for k, p in self.model_hf.named_parameters():
                p.requires_grad = False


        for i  in range(4):
            setattr(self, f"extra_pat_{i+1}", ExtractionPattern(dim = self.prompt_dim, dim_dino=self.token_dim, depth=self.ext_num))

            setattr(self, f"down_pat_{i + 1}", nn.Sequential(
                nn.Linear(self.token_dim, self.prompt_dim),
                nn.GELU(),
                nn.Linear(self.prompt_dim, self.prompt_dim)
            ))
            setattr(self, f"down_feat_{i + 1}", nn.Sequential(
                nn.Linear(self.token_dim, self.out_dim),
                nn.GELU(),
                nn.Linear(self.out_dim, self.out_dim)
            ))

    def insert(self, prompt, hidden_state):
        self.extra_pat(prompt, hidden_state)

    def make_mask(self, n_token):
        mask = torch.zeros(1, 1, n_token+self.num_prompt, n_token+self.num_prompt)
        mask[:, :, :self.num_prompt,:self.num_prompt] = -1e9
        return mask.cuda()

    def select_mask(self, dataset):
        if dataset in ["NYUDataset", "SUNRGBDDataset", "iBimsDataset"]:
            return self.make_mask(1531)
        if dataset == "KITTIDataset":
            return self.make_mask(2151)
        if dataset in ["DDADDataset", "ArgoverseDataset"]:
            return self.make_mask(3732)


    def make_mask_all_dateset(self, config):
        train_name = config["data"]["train_dataset"]["name"]
        setattr(self,f"mask_{train_name}",self.select_mask(train_name))
        valsets = config["data"]["val_dataset"]
        for _val in valsets:
            val_name = _val["name"]
            setattr(self, f"mask_{val_name}", self.select_mask(val_name))


    def forward(self, x, feat, mod):
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            H_new = H // self.patch_size * self.patch_size
            W_new = W // self.patch_size * self.patch_size
            x = nn.functional.interpolate(x, size=(H_new, W_new), mode='bilinear', align_corners=False)
        else:
            H_new = H
            W_new = W

        prompt1 = self.prompt_embeddings_1.expand(B, -1, -1)
        prompt2 = self.prompt_embeddings_2.expand(B, -1, -1)
        prompt3 = self.prompt_embeddings_3.expand(B, -1, -1)
        prompt4 = self.prompt_embeddings_4.expand(B, -1, -1)

        prompt1 = torch.fft.fft(torch.fft.fft(prompt1, dim=-1), dim=-2).real
        prompt2 = torch.fft.fft(torch.fft.fft(prompt2, dim=-1), dim=-2).real
        prompt3 = torch.fft.fft(torch.fft.fft(prompt3, dim=-1), dim=-2).real
        prompt4 = torch.fft.fft(torch.fft.fft(prompt4, dim=-1), dim=-2).real

        x = self.model_hf.embeddings(x)


        mask = getattr(self, f"mask_{mod}").expand(B, -1, -1, -1)

        f1 = self.extra_pat_1(prompt1, feat[0])
        x = torch.cat((f1,x), dim=1)
        x = self.model_hf.encoder.stage1(x, self.layer_idx, mask)
        pat1 = x[:,:self.num_prompt,:]
        pat1_down = self.down_pat_1(pat1)
        feat1_down = self.down_feat_1(x[:,self.num_prompt+1:,:])

        f2 = self.extra_pat_2(prompt2 + pat1_down, feat[1])
        x = torch.cat((f2, x[:, self.num_prompt:, :]), dim=1)
        x = self.model_hf.encoder.stage2(x, self.layer_idx, mask)
        pat2 = x[:,:self.num_prompt,:]
        pat2_down = self.down_pat_2(pat2)
        feat2_down = self.down_feat_2(x[:,self.num_prompt+1:,:])

        f3 = self.extra_pat_3(prompt3 + pat2_down, feat[2])
        x = torch.cat((f3, x[:, self.num_prompt:, :]), dim=1)
        x = self.model_hf.encoder.stage3(x, self.layer_idx, mask)
        pat3 = x[:,:self.num_prompt,:]
        pat3_down = self.down_pat_3(pat3)
        feat3_down = self.down_feat_3(x[:, self.num_prompt + 1:, :])

        f4 = self.extra_pat_4(prompt4 + pat3_down, feat[3])
        x = torch.cat((f4, x[:, self.num_prompt:, :]), dim=1)
        x = self.model_hf.encoder.stage4(x, self.layer_idx, mask)
        pat4 = x[:, :self.num_prompt, :]
        pat4_down = self.down_pat_4(pat4)
        feat4_down = self.down_feat_4(x[:, self.num_prompt + 1:, :])

        pat_all = [pat1_down, pat2_down, pat3_down, pat4_down]
        vit_feat_all = [feat1_down, feat2_down, feat3_down, feat4_down]

        return pat_all, vit_feat_all



import math
import torch
import torch.nn as nn
from einops import rearrange
from .Attention import PerceiverSelfAttention, MLP
from .pos_emb import PositionEmbeddingSine


class ExtractionPattern(nn.Module):
    def __init__(self, dim, dim_dino, init_values=1, depth=2):
        super().__init__()
        self.iters = depth
        for i in range(self.iters):

            setattr(self, f"cross_attn_{i+1}", PerceiverSelfAttention(q_dim=dim, kv_dim=dim, qk_channels=dim, v_channels=dim, num_heads=8))
            setattr(self, f"self_attn_{i + 1}", PerceiverSelfAttention(is_cross_attention=False, q_dim=dim, kv_dim=dim, qk_channels=dim, v_channels=dim, num_heads=8))
            setattr(self, f"gamma_{i+1}", nn.Parameter(init_values * torch.ones((dim)), requires_grad=True))

        # setattr(self,f"proj", nn.Linear(dim, dim_dino))
        setattr(self, f"proj", nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim_dino)
        ))



    def forward(self, pat, feat):

        feat = rearrange(feat, "b c h w -> b (h w) c")
        for i in range(self.iters):
            pat = pat + getattr(self,f"gamma_{i+1}") * getattr(self, f"cross_attn_{i+1}")(hidden_states=pat, inputs=feat)['context']
            pat = pat + getattr(self, f"self_attn_{i+1}")(hidden_states=pat, inputs=pat)['context']

        pat = getattr(self, f"proj")(pat) # 与dino token 维度对齐

        return pat



class MergeFeatureBackboneHead(nn.Module):
    def __init__(self, n_layer, dim):
        super().__init__()
        self.n_CossAtt_dec = n_layer
        self.pixel_pe = PositionEmbeddingSine(dim // 2, normalize=True)


        setattr(self, f"self_attn_enc",
                PerceiverSelfAttention(is_cross_attention=False, q_dim=dim, kv_dim=dim, qk_channels=dim, v_channels=dim,
                                       num_heads=8))
        for i in range(self.n_CossAtt_dec):

            setattr(self, f"cross_attn_enc_{i+1}", PerceiverSelfAttention(q_dim=dim, kv_dim=dim, qk_channels=dim, v_channels=dim, num_heads=8))
            setattr(self, f"mlp_attn_enc_{i+1}", MLP(dim))


    def forward(self, x_e, lat):
        b, c, h, w = x_e.shape
        x_e = x_e + self.pixel_pe(x_e)
        x_e = rearrange(x_e, "b c h w -> b (h w) c").contiguous()
        # lat =self.project(lat)
        # lat = self.self_attn_enc(hidden_states=lat)['context']
        lat = lat + getattr(self, f"self_attn_enc")(hidden_states=lat, inputs=lat)['context']
        for i in range(self.n_CossAtt_dec):
            x_e = x_e + getattr(self, f"cross_attn_enc_{i+1}")(hidden_states=x_e, inputs=lat)['context']
            x_e = x_e + getattr(self, f"mlp_attn_enc_{i+1}")(x_e)
        # out = getattr(self,f"proj_output")(x_e)
        x_e = rearrange(x_e, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        return x_e


class ProjectOut(nn.Module):
    def __init__(self, input=256):
        super().__init__()
        # self.max_depth = cfg.max_depth
        self.num_resolutions = 4

        for i in range(self.num_resolutions):
            setattr(self, f"proj_output_{i+1}", nn.Sequential(nn.BatchNorm2d(input),
                                                    nn.Conv2d(input, input//2,3,1, 1),
                                                    nn.ConvTranspose2d(input//2, input//2, 2,2,0),
                                                    nn.Conv2d(input//2, input//2, 3, 1, 1),
                                                    nn.BatchNorm2d(input//2),
                                                    nn.Conv2d(input//2, 1,1,1),
                                                    ))

            nn.init.kaiming_normal_(getattr(self, f"proj_output_{i + 1}")[1].weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(getattr(self, f"proj_output_{i + 1}")[2].weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(getattr(self, f"proj_output_{i + 1}")[3].weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(getattr(self, f"proj_output_{i + 1}")[5].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x_e):
        outs = ()
        for i in range(self.num_resolutions):
            out = getattr(self, f"proj_output_{i+1}")(x_e[i])
            outs = outs + (out,)
        return outs


class MergeFeatureBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cusum = cfg.cusum
        self.num_resolutions = 4
        self.num_trans = [0,1,2,3]
        for i in range(self.num_resolutions):
            if i in self.num_trans:
                setattr(self, f"head_{i + 1}", MergeFeatureBackboneHead(cfg.n_CossAtt_dec, cfg.lat_dim))
                setattr(self, f"proj_{i + 1}", ProjectOut(cfg))
            else:
                setattr(self, f"proj_{i + 1}", ProjectOut(cfg))



    def forward(self, x, lat):
        outs = ()
        # print(f"length_xfpn:{len(x)}")
        for i in range(self.num_resolutions):
            # b, c, h, w = x[i].shape
            # x_t = rearrange(x[i], "b c h w -> b (h w) c")
            # x_t = x[i]
            if i in self.num_trans:
                if self.cusum:
                    out = getattr(self, f"head_{i + 1}")(x[i], torch.cat(lat[:i+1], dim=1))
                else:
                    out = getattr(self, f"head_{i+1}")(x[i], lat[i])
                # out = torch.cat((out,x_t),dim=1)
                out = getattr(self, f"proj_{i + 1}")(out)


            else:
                out = getattr(self, f"proj_{i + 1}")(x[i])

            # out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
            outs = outs + (out,)

        return outs


class MergeFeaturePattern(nn.Module):
    def __init__(self, inj_num, dim):
        super().__init__()
        self.cusum = False
        self.num_resolutions = 4
        self.num_trans = [0,1,2,3]
        for i in range(self.num_resolutions):
            if i in self.num_trans:
                setattr(self, f"head_{i + 1}", MergeFeatureBackboneHead(inj_num, dim))


    def forward(self, x, lat):

        outs = ()

        for i in range(self.num_resolutions):

            if i in self.num_trans:
                if self.cusum:
                    out = getattr(self, f"head_{i + 1}")(x[i], torch.cat(lat[:i+1], dim=1))
                else:
                    out = getattr(self, f"head_{i+1}")(x[i], lat[i])

                outs = outs + (out,)

        return outs



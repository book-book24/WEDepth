import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
    Resize,
)
from .VFM import PromptDINOV2

from module import encoder
from .merge import MergeFeaturePattern, ProjectOut
from .decoder import PixelDecoder
from .defattn_decoder import MSDeformAttnPixelDecoder
from .neck import SplitFeature, FuseFuture

class WEDepth(nn.Module):

    def __init__(self, config):
        super().__init__()
        prompt_dim = config["model"]["prompt_dim"]

        self.encoder = getattr(encoder, config["model"]["encoder_name"])()
        self.sf = SplitFeature(prompt_dim, self.backbone.embed_dims)
        self.partition_enhance = PromptDINOV2(config)
        self.inject_pt = MergeFeaturePattern(config["model"]["inj_num"], prompt_dim)
        self.inject_it = FuseFuture(prompt_dim, config["model"]["patch_size"])
        self.decoder = MSDeformAttnPixelDecoder(input_dims=[prompt_dim]*4, dropout=config["model"]["dropout"], num_heads=4,  hidden_dim=256, depth=4, output_dim=256, ffn_dim=256*2,activation='gelu')
        self.project = ProjectOut(input=256)

    def forward(self,x, gt, mod):

        img_wh = x.shape[-2:]
        original_wh = gt.shape[-2:]
        x_all = self.encoder(x)
        x_b1, x_b2 = self.sf(x_all)
        pat_all, x_s = self.partition_enhance(x, x_b2, mod)
        x_b1 = self.inject(x_b1, pat_all)
        x_merge = self.fuse(x_b1, x_s, img_wh)
        outs, _ = self.decoder(x_merge[::-1])
        outs = self.project(outs)
        out_list = []
        for out in outs:
            out =  F.interpolate(torch.exp(out), size=outs[-1].shape[-2:], mode="bilinear", align_corners=True)
            out_list.append(out)

        pred = F.interpolate(
            torch.mean(torch.stack(out_list, dim=0), dim=0),
            size=original_wh, mode="bilinear", align_corners=True)

        return pred

    @torch.no_grad()
    def infer(self, image, mod):
        image = self.image2tensor(image)
        image_f = torch.flip(image,[3])

        depth1, _ = self.forward(image, image, mod)
        depth2, _ = self.forward(image_f, image_f, mod)
        depth2 = torch.flip(depth2, [3])
        depth = 0.5 * (depth1 + depth2)

        return depth.squeeze().cpu().numpy()


    def image2tensor(self, image, input_size=518):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        transform = Compose(
            [
                ToTensor(),
                # Resize(input_size),
                Lambda(lambda x: x.to(DEVICE)),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ConvertImageDtype(torch.float32),
            ]
        )

        image = transform(image)

        return image.unsqueeze(0)




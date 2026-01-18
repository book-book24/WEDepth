import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union

class PixelDecoder(nn.Module):
    def __init__(self,  hidden_dim = 256, output_dim = 256,):
        super().__init__()
        # self.in_features = np.array([64, 128, 256, 512])*4
        # self.in_features = [128 * (2 ** i) for i in range(4)]
        self.in_features = [384, 384, 384, 384]
        # self.in_features = [256, 256, 256, 256]
        self.in_features = self.in_features[::-1]
        for idx, in_channels in enumerate(self.in_features):
            if idx == 0:
                output_conv = nn.Sequential(
                    nn.Conv2d(in_channels, output_dim, 3, 1, 1),
                    nn.BatchNorm2d(output_dim),
                    nn.SiLU())
                nn.init.kaiming_uniform_(output_conv[0].weight, a=1)
                nn.init.constant_(output_conv[0].bias, 0)
                self.add_module(f"layer_{idx + 1}", output_conv)
            else:
                lateral_conv = nn.Sequential(
                    nn.Conv2d(in_channels, output_dim, 1, 1, 0),
                    nn.BatchNorm2d(output_dim),
                    nn.SiLU())
                output_conv = nn.Sequential(
                    nn.Conv2d(output_dim, output_dim, 3, 1, 1),
                    nn.BatchNorm2d(output_dim),
                    nn.SiLU())

                nn.init.kaiming_uniform_(lateral_conv[0].weight, a=1)
                nn.init.constant_(lateral_conv[0].bias, 0)
                nn.init.kaiming_uniform_(output_conv[0].weight, a=1)
                nn.init.constant_(output_conv[0].bias, 0)

                self.add_module(f"adapter_{idx + 1}", lateral_conv)
                self.add_module(f"layer_{idx + 1}", output_conv)


    def forward(self, x):
        # x = x[1:]
        x = x[::-1]
        fpn_output = []
        fpn_shape = []
        for idx, f in enumerate(x):
            # print(f"idx:{idx}")
            xd = x[idx]
            fpn_shape.append(xd.shape)
            if idx == 0:
                y = getattr(self,f"layer_{idx + 1}")(xd)
            else:
                cur = getattr(self, f"adapter_{idx + 1}")(xd)
                y = cur + F.interpolate(y, size=cur.shape[-2:], mode="nearest")
                y = getattr(self, f"layer_{idx + 1}")(y)
            fpn_output.append(y)


        return fpn_output, fpn_shape


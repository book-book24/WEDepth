import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from openpyxl.styles.builtins import output


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.F_squeeze = nn.AdaptiveAvgPool2d(1)
        self.F_excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        B, C, H, W = x.size()
        y = self.F_squeeze(x).view(B, C).contiguous()
        y = self.F_excitation(y).view(B, C, 1, 1).contiguous()
        output = x * y
        return output



class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        d = dim // 2
        d1 = dim
        self.C = d1
        self.in_features = [128 * (2 ** i) for i in range(4)]
        self.fc1 = nn.Conv2d(self.in_features[0], d1, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.in_features[1], d1, 1, 1, 0)
        self.fc3 = nn.Conv2d(self.in_features[2], d1, 1, 1, 0)
        self.fc4 = nn.Conv2d(self.in_features[3], d1, 1, 1, 0)

        self.dwconv1_1 = nn.Conv2d(d, d, 3, 1, 1, bias=True, groups=d)
        self.dwconv1_2 = nn.Conv2d(d, d, 5, 1, 2, bias=True, groups=d)

        self.dwconv2_1 = nn.Conv2d(d, d, 3, 1, 1, bias=True, groups=d)
        self.dwconv2_2 = nn.Conv2d(d, d, 5, 1, 2, bias=True, groups=d)

        self.dwconv3_1 = nn.Conv2d(d, d, 3, 1, 1, bias=True, groups=d)
        self.dwconv3_2 = nn.Conv2d(d, d, 5, 1, 2, bias=True, groups=d)

        self.dwconv4_1 = nn.Conv2d(d, d, 3, 1, 1, bias=True, groups=d)
        self.dwconv4_2 = nn.Conv2d(d, d, 5, 1, 2, bias=True, groups=d)

        self.act1 = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(d1)

        self.act2 = nn.SiLU()
        self.norm2 = nn.BatchNorm2d(d1)

        self.act3 = nn.SiLU()
        self.norm3 = nn.BatchNorm2d(d1)

        self.act4 = nn.SiLU()
        self.norm4 = nn.BatchNorm2d(d1)


    def forward(self, x):

        x1, x2, x3, x4 = x

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        x4 = self.fc4(x4)

        x11, x12 = x1[:, :self.C // 2, :, :], x1[:, self.C // 2:, :, :]
        x11 = self.dwconv1_1(x11) # BxCxHxW
        x12 = self.dwconv1_2(x12)
        x1 = torch.cat([x11, x12], dim=1)
        # x1 = self.act1(self.norm1(x1))

        x21, x22 = x2[:, :self.C // 2, :, :], x2[:, self.C // 2:, :, :]
        x21 = self.dwconv2_1(x21) # BxCxHxW
        x22 = self.dwconv2_2(x22)
        x2 = torch.cat([x21, x22], dim=1)
        # x2 = self.act2(self.norm2(x2))

        x31, x32 = x3[:, :self.C // 2, :, :], x3[:, self.C // 2:, :, :]
        x31 = self.dwconv1_1(x31) # BxCxHxW
        x32 = self.dwconv1_2(x32)
        x3 = torch.cat([x31, x32], dim=1)
        # x3 = self.act3(self.norm3(x3))

        x41, x42 = x4[:, :self.C // 2, :, :], x4[:, self.C // 2:, :, :]
        x41 = self.dwconv4_1(x41) # BxCxHxW
        x42 = self.dwconv4_2(x42)
        x4 = torch.cat([x41, x42], dim=1)
        # x4 = self.act4(self.norm4(x4))

        return [x1, x2, x3, x4]

class DWConv1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        d = dim // 2
        d1 = dim
        self.C = d


        self.dwconv1_1 = nn.Conv2d(d, d, 3, 1, 1, bias=True, groups=d)
        self.dwconv1_2 = nn.Conv2d(d, d, 5, 1, 2, bias=True, groups=d)

        self.dwconv2_1 = nn.Conv2d(d, d, 3, 1, 1, bias=True, groups=d)
        self.dwconv2_2 = nn.Conv2d(d, d, 5, 1, 2, bias=True, groups=d)

        self.dwconv3_1 = nn.Conv2d(d, d, 3, 1, 1, bias=True, groups=d)
        self.dwconv3_2 = nn.Conv2d(d, d, 5, 1, 2, bias=True, groups=d)

        self.dwconv4_1 = nn.Conv2d(d, d, 3, 1, 1, bias=True, groups=d)
        self.dwconv4_2 = nn.Conv2d(d, d, 5, 1, 2, bias=True, groups=d)

        self.act1 = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(d1)

        self.act2 = nn.SiLU()
        self.norm2 = nn.BatchNorm2d(d1)

        self.act3 = nn.SiLU()
        self.norm3 = nn.BatchNorm2d(d1)

        self.act4 = nn.SiLU()
        self.norm4 = nn.BatchNorm2d(d1)


    def forward(self, x):

        x1, x2, x3, x4 = x


        x11, x12 = x1[:, :self.C, :, :], x1[:, self.C:, :, :]
        x11 = self.dwconv1_1(x11) # BxCxHxW
        x12 = self.dwconv1_2(x12)
        x1 = torch.cat([x11, x12], dim=1)
        # x1 = self.act1(self.norm1(x1))

        x21, x22 = x2[:, :self.C, :, :], x2[:, self.C:, :, :]
        x21 = self.dwconv2_1(x21) # BxCxHxW
        x22 = self.dwconv2_2(x22)
        x2 = torch.cat([x21, x22], dim=1)
        # x2 = self.act2(self.norm2(x2))

        x31, x32 = x3[:, :self.C, :, :], x3[:, self.C:, :, :]
        x31 = self.dwconv1_1(x31) # BxCxHxW
        x32 = self.dwconv1_2(x32)
        x3 = torch.cat([x31, x32], dim=1)
        # x3 = self.act3(self.norm3(x3))

        x41, x42 = x4[:, :self.C, :, :], x4[:, self.C:, :, :]
        x41 = self.dwconv4_1(x41) # BxCxHxW
        x42 = self.dwconv4_2(x42)
        x4 = torch.cat([x41, x42], dim=1)
        # x4 = self.act4(self.norm4(x4))

        x_out = [x1, x2, x3, x4]

        return x_out


class SplitFeature(nn.Module):
    def __init__(self, dim_token, in_dim):
        super().__init__()
        self.d = dim_token

        self.in_features = in_dim

        self.fc1 = nn.Conv2d(self.in_features[0], self.d, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.in_features[1], self.d, 1, 1, 0)
        self.fc3 = nn.Conv2d(self.in_features[2], self.d, 1, 1, 0)
        self.fc4 = nn.Conv2d(self.in_features[3], self.d, 1, 1, 0)


        self.dwconv1 = nn.Conv2d(self.d, self.d, 3, 1, 1, bias=True, groups=self.d)
        self.dwconv2 = nn.Conv2d(self.d, self.d, 3, 1, 1, bias=True, groups=self.d)
        self.dwconv3 = nn.Conv2d(self.d, self.d, 3, 1, 1, bias=True, groups=self.d)
        self.dwconv4 = nn.Conv2d(self.d, self.d, 3, 1, 1, bias=True, groups=self.d)


    def forward(self, x):
        x1, x2, x3, x4 = x
        xc1 = self.fc1(x1)
        xc2 = self.fc2(x2)
        xc3 = self.fc3(x3)
        xc4 = self.fc4(x4)

        x_c = [xc1, xc2, xc3, xc4]

        xs1 = self.dwconv1(xc1)
        xs2 = self.dwconv1(xc2)
        xs3 = self.dwconv1(xc3)
        xs4 = self.dwconv1(xc4)

        x_s = [xs1, xs2, xs3, xs4]


        return x_s, x_c



class FuseFuture(nn.Module):
    def __init__(self, dim, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        for i in range(4):
            setattr(self,f"conv_fuse_{i+1}", nn.Conv2d(dim, dim, 3,1,1, groups=dim))
            setattr(self, f"liner_{i + 1}", nn.Conv2d(dim, dim, 1, 1, 0))
            setattr(self, f"conv_fuse_v_{i + 1}", nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
            setattr(self, f"norm_{i + 1}", nn.BatchNorm2d(dim))
            setattr(self, f"act_{i + 1}", nn.ReLU())
        self.conv_all = nn.Conv2d(dim, dim, 1, 1,0)

    def forward(self,x_b1, vit_feat, hw):
        assert self.patch_size == 14

        x_all = []
        h = hw[0] // self.patch_size
        w = hw[1] // self.patch_size

        x_s1, x_s2, x_s3, x_s4 = vit_feat


        x_s4 = rearrange(x_s4, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        x_s1 = F.interpolate(x_s4, size=(math.ceil(hw[0] / 4), math.ceil(hw[1] / 4)), mode='bilinear', align_corners=False)
        x_s2 = F.interpolate(x_s4, size=(math.ceil(hw[0] / 8), math.ceil(hw[1] / 8)), mode='bilinear', align_corners=False)
        x_s3 = F.interpolate(x_s4, size=(math.ceil(hw[0] / 16), math.ceil(hw[1] / 16)), mode='bilinear', align_corners=False)
        x_s4 = F.interpolate(x_s4, size=(math.ceil(hw[0] / 32), math.ceil(hw[1] / 32)), mode='bilinear', align_corners=False)

        xs_all = [x_s1, x_s2, x_s3, x_s4]

        for i in range(4):

            xc = x_b1[i]
            xs = xs_all[i]
            xc = xc + getattr(self,f"conv_fuse_{i+1}")(xc)
            x = xc + getattr(self,f"liner_{i+1}")(xs)
            x = x + getattr(self, f"conv_fuse_v_{i + 1}")(x)
            x = getattr(self, f"norm_{i + 1}")(x)
            x = getattr(self, f"act_{i + 1}")(x)
            x = self.conv_all(x)
            x_all.append(x)

        return x_all


class FuseFutureS(nn.Module):
    def __init__(self, dim):
        super().__init__()
        for i in range(4):
            setattr(self,f"conv_fuse_{i+1}", nn.Conv2d(dim, dim, 3,1,1, groups=dim))
            setattr(self, f"proj_{i+1}", nn.Conv2d(dim, dim, 1, 1, 0))
            # setattr(self,f"norm_{i+1}", nn.LayerNorm(dim))
            setattr(self, f"proj_s_{i + 1}", nn.Conv2d(dim*2, 1, 1, 1, 0))
            setattr(self, f"norm_{i + 1}", nn.BatchNorm2d(dim))

    def forward(self,x1, x2, vit_feat, hw):
        x_all = []
        h = hw[0]//16
        w = hw[1]//16
        for i in range(4):
            # x = torch.cat((x1[i], x2[i]), dim=1)
            x = x1[i] + x2[i]
            x = x + getattr(self,f"conv_fuse_{i+1}")(x)
            x = getattr(self, f"proj_{i+1}")(x)
            # x = x.permute(0, 2, 3, 1)
            # x = getattr(self, f"norm_{i+1}")(x)
            # x = x.permute(0, 3, 1, 2).contiguous()
            x_all.append(x)

        x_s1, x_s2, x_s3, x_s4 = vit_feat

        x_s1 = rearrange(x_s1, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x_s2 = rearrange(x_s2, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x_s3 = rearrange(x_s3, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x_s4 = rearrange(x_s4, "b (h w) c -> b c h w", h=h, w=w).contiguous()

        x_s1 = self.proj_s_1(x_s1)
        x_s2 = self.proj_s_2(x_s2)
        x_s3 = self.proj_s_3(x_s3)
        x_s4 = self.proj_s_4(x_s4)

        x_s1 = F.interpolate(x_s1, scale_factor=4, mode='bilinear', align_corners=False)
        x_s2 = F.interpolate(x_s2, scale_factor=2, mode='bilinear', align_corners=False)
        x_s4 = F.interpolate(x_s4, scale_factor=0.5, mode='bilinear', align_corners=False)

        x_all[0] = self.norm_1(x_all[0] * F.sigmoid(x_s1))
        x_all[1] = self.norm_2(x_all[1] * F.sigmoid(x_s2))
        x_all[2] = self.norm_3(x_all[2] * F.sigmoid(x_s3))
        x_all[3] = self.norm_4(x_all[3] * F.sigmoid(x_s4))

        return x_all


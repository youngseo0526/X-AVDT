import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize


class ProjectionHead(nn.Module):
    """Projection head for the (triplet) contrastive embedding.

    NOTE: the released checkpoint stores the inner Sequential weights under
    indices `proj.0` and `proj.2` (i.e. a parameter-free layer sits at index 1).
    We therefore use LayerNorm -> GELU -> Linear so the state_dict keys are
    `proj.0.{weight,bias}` (LayerNorm) and `proj.2.{weight,bias}` (Linear),
    matching `img_proj.*` / `attn_proj.*` in the checkpoint.
    """
    def __init__(self, in_dim=1024, proj_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),   # proj.0
            nn.GELU(),              # proj.1 (no params -> index 1 skipped in state_dict)
            nn.Linear(in_dim, proj_dim),  # proj.2
        )

    def forward(self, x):
        return self.proj(x)


def get_norm(norm_type, num_features, dim=3):
    if norm_type == "batch":
        return nn.BatchNorm3d(num_features)
    elif norm_type == "instance":
        return nn.InstanceNorm3d(num_features, affine=True)
    elif norm_type == "layer":
        return nn.GroupNorm(1, num_features)
    else:
        raise ValueError(f"Unsupported norm layer: {norm_type}")


class ResNeXtBottleneck3D(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, norm_type="batch"):
        super().__init__()
        D = cardinality * base_width
        self.conv_reduce = nn.Conv3d(in_channels, D, kernel_size=1, bias=False)
        self.bn_reduce   = get_norm(norm_type, D)
        self.conv_group  = nn.Conv3d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn_group    = get_norm(norm_type, D)
        self.conv_expand = nn.Conv3d(D, out_channels, kernel_size=1, bias=False)
        self.bn_expand   = get_norm(norm_type, out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                get_norm(norm_type, out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv_reduce(x)
        out = self.bn_reduce(out)
        out = self.relu(out)
        out = self.conv_group(out)
        out = self.bn_group(out)
        out = self.relu(out)
        out = self.conv_expand(out)
        out = self.bn_expand(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNeXt_imgencoder(nn.Module):
    def __init__(self, block, layers, is_res_feat=True, cardinality=32, base_width=4, norm_type="batch"):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(12 if is_res_feat else 9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = get_norm(norm_type, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, cardinality=cardinality, base_width=base_width, norm_type=norm_type)
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2, cardinality=cardinality, base_width=base_width*2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=2, cardinality=cardinality, base_width=base_width*4, norm_type=norm_type)

    def _make_layer(self, block, out_channels, blocks, stride, cardinality, base_width, norm_type):
        layers = [block(self.in_channels, out_channels, stride, cardinality, base_width, norm_type)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, cardinality, base_width, norm_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def get_img_encoder(is_res_feat=True, norm_type="batch"):
    return ResNeXt_imgencoder(ResNeXtBottleneck3D, [2, 3, 3], cardinality=32, base_width=4, is_res_feat=is_res_feat, norm_type=norm_type)


class ResNeXt_attnfeatencoder(nn.Module):
    def __init__(self, block, layers, cardinality=32, base_width=4, norm_type="batch", in_channels_2d=320):
        super().__init__()
        self.expected_c = in_channels_2d
        self.in_channels = 512
        self.conv1 = nn.Conv3d(in_channels_2d, 512, kernel_size=3, stride=(2,1,1), padding=1, bias=False)
        self.bn1   = get_norm(norm_type, 512)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,  512, layers[0], stride=(2,1,1), cardinality=cardinality, base_width=base_width*2, norm_type=norm_type)
        self.layer2 = self._make_layer(block,  512, layers[1], stride=(2,1,1), cardinality=cardinality, base_width=base_width*2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, 1024, layers[2], stride=2,        cardinality=cardinality, base_width=base_width*4, norm_type=norm_type)

    def _make_layer(self, block, out_channels, blocks, stride, cardinality, base_width, norm_type):
        layers = [block(self.in_channels, out_channels, stride, cardinality, base_width, norm_type)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, cardinality, base_width, norm_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 5, f"attn_feat must be 5D (B,C,T,H,W), got {x.shape}"
        assert x.size(1) == self.expected_c, f"attn_feat C={x.size(1)} expected {self.expected_c}"
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def get_feat_encoder(is_res_feat=True, norm_type="batch", in_channels_2d=320):
    # Checkpoint `sa_triplet_dec_bs8_800_es.pt` has only layer3.{0,1}
    # (no layer3.2 / layer3.3), i.e. layer3 holds 2 blocks -> [2, 2, 2].
    return ResNeXt_attnfeatencoder(ResNeXtBottleneck3D, [2, 2, 2], cardinality=32, base_width=4, norm_type=norm_type, in_channels_2d=in_channels_2d)


class Fuse_Attn_Decoder(nn.Module):
    def __init__(self, block, layers, cardinality=32, base_width=4, self_attn=True):
        super().__init__()
        self.self_attn = self_attn
        self.embed_dim = 1024
        self.channel_proj = nn.Conv2d(2048, self.embed_dim, kernel_size=1)
        if self.self_attn:
            # Checkpoint stores a single `decoder.layernorm` (no `attn_layernorm`,
            # no `post_ln`); it is reused for the pre-norm and the post-residual norm.
            self.attention_fuse = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)
            self.linear = nn.Linear(self.embed_dim, self.embed_dim)
            self.layernorm = nn.LayerNorm(self.embed_dim)
            self.relu = nn.ReLU(inplace=True)
        self.in_channels = self.embed_dim
        self.layer1 = self._make_layer(block, self.embed_dim, layers[0], stride=2, cardinality=cardinality, base_width=base_width*8)
        self.layer2 = self._make_layer(block, self.embed_dim, layers[1], stride=2, cardinality=cardinality, base_width=base_width*8)
        self.layer3 = self._make_layer(block, self.embed_dim, layers[2], stride=2, cardinality=cardinality, base_width=base_width*8)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Checkpoint head is (2, 1024): 2-class softmax classifier.
        self.fc = nn.Linear(self.embed_dim, 2)

    def _make_layer(self, block, out_channels, blocks, stride, cardinality, base_width):
        layers = [block(self.in_channels, out_channels, stride, cardinality, base_width)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, cardinality, base_width))
        return nn.Sequential(*layers)

    def get_2d_sincos_pos_embed(self, C, H, W, device=None, dtype=torch.float32):
        if C % 4 != 0:
            raise ValueError(f"C must be divisible by 4, got {C}")
        C_quarter = C // 4
        dim_range = torch.arange(C_quarter, dtype=dtype, device=device)
        inv_freq = 1.0 / (10000 ** (dim_range / C_quarter))
        pos_h = torch.arange(H, dtype=dtype, device=device).unsqueeze(1)
        pos_w = torch.arange(W, dtype=dtype, device=device).unsqueeze(1)
        ang_h = pos_h * inv_freq.unsqueeze(0)
        ang_w = pos_w * inv_freq.unsqueeze(0)
        pe_h = torch.cat([torch.sin(ang_h), torch.cos(ang_h)], dim=1).transpose(0, 1)
        pe_w = torch.cat([torch.sin(ang_w), torch.cos(ang_w)], dim=1).transpose(0, 1)
        pe = torch.zeros(C, H, W, dtype=dtype, device=device)
        pe[:2*C_quarter, :, :] = pe_h.unsqueeze(-1).repeat(1, 1, W)
        pe[2*C_quarter:, :, :] = pe_w.unsqueeze(1).repeat(1, H, 1)
        return pe

    def add_2d_positional_encoding(self, x):
        B, C, H, W = x.shape
        pe = self.get_2d_sincos_pos_embed(C, H, W, device=x.device, dtype=x.dtype)
        pe = pe.unsqueeze(0)
        return x + pe

    def forward(self, img_feat, attn_feat=None):
        if attn_feat is None:
            x = img_feat.squeeze(2)  # (B,C,H,W)
        else:
            img_feat = img_feat.squeeze(2)
            attn_feat = attn_feat.squeeze(2)
            x = torch.cat((img_feat, attn_feat), dim=1)  # (B, 2048, H, W)
            x = self.channel_proj(x)                     # (B, 1024, H, W)

        if self.self_attn and attn_feat is not None:
            x_pe = self.add_2d_positional_encoding(x)
            B, C, H, W = x_pe.shape
            x_flat = x_pe.flatten(2).transpose(1, 2)     # (B, HW, C)
            x_norm = self.layernorm(x_flat)
            x_attn = self.attention_fuse(x_norm, x_norm, x_norm)[0]
            x_attn = self.linear(x_attn)
            x_attn = self.layernorm(x_attn + x_flat)
            x_attn = self.relu(x_attn)
            x = x_attn.transpose(1, 2).view(B, C, H, W)

        x = x.unsqueeze(2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        embed = x
        x_out = self.fc(x)  # (B, 2)
        return x_out, embed


class Fuse_Decoder(nn.Module):
    def __init__(self, block, layers, cardinality=32, base_width=4):
        super().__init__()
        self.in_channels = 1024
        self.layer1 = self._make_layer(block, 1024, layers[0], stride=2, cardinality=cardinality, base_width=base_width*4)
        self.layer2 = self._make_layer(block, 2048, layers[1], stride=2, cardinality=cardinality, base_width=base_width*8)
        self.layer3 = self._make_layer(block, 2048, layers[2], stride=2, cardinality=cardinality, base_width=base_width*8)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, 2)

    def _make_layer(self, block, out_channels, blocks, stride, cardinality, base_width):
        layers = [block(self.in_channels, out_channels, stride, cardinality, base_width)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, cardinality, base_width))
        return nn.Sequential(*layers)

    def forward(self, input_feat):
        x = input_feat.squeeze(2)
        x = x.unsqueeze(2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embed = x
        x = self.fc(x)  # (B, 2)
        return x, embed


def get_feat_decoder(is_fuse=True, is_self_attn=True):
    if is_fuse:
        if is_self_attn:
            return Fuse_Attn_Decoder(ResNeXtBottleneck3D, [2, 2, 2], cardinality=32, base_width=4, self_attn=True)
        else:
            return Fuse_Attn_Decoder(ResNeXtBottleneck3D, [2, 2, 2], cardinality=32, base_width=4, self_attn=False)
    else:
        return Fuse_Decoder(ResNeXtBottleneck3D, [3, 2, 2], cardinality=32, base_width=4)


class Classifier(nn.Module):
    def __init__(
        self,
        self_attn=True,
        norm_layer="batch",
        attn_in_channels=320,
        proj_dim=512,
    ):
        super().__init__()

        self.img_encoder = get_img_encoder(norm_type=norm_layer)  # -> (B,1024,T,H,W)
        self.attn_encoder = get_feat_encoder(norm_type=norm_layer, in_channels_2d=attn_in_channels)  # -> (B,1024,T,H,W)
        self.decoder = get_feat_decoder(is_fuse=True, is_self_attn=self_attn)

        # Triplet / contrastive projection heads (present in the released checkpoint
        # as `img_proj.*` and `attn_proj.*`). Each head: 1024 -> proj_dim.
        # The two projected vectors are concatenated to form the (2*proj_dim)-d
        # embedding used by the contrastive loss (default 512*2 = 1024 == embedding_size).
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.img_proj = ProjectionHead(in_dim=1024, proj_dim=proj_dim)
        self.attn_proj = ProjectionHead(in_dim=1024, proj_dim=proj_dim)

    def forward(self, x, attn_feat):
        img_feat = self.img_encoder(x)
        attn_feat_encoded = self.attn_encoder(attn_feat)
        out, _ = self.decoder(img_feat, attn_feat_encoded)

        vi = self.pool(img_feat).flatten(1)            # (B, 1024)
        va = self.pool(attn_feat_encoded).flatten(1)   # (B, 1024)
        ei = self.img_proj(vi)                         # (B, proj_dim)
        ea = self.attn_proj(va)                        # (B, proj_dim)
        embed = torch.cat([ei, ea], dim=1)             # (B, 2*proj_dim)

        return out, embed

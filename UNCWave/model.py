RUN_TRAIN = True # bfloat16 or float32 recommended
RUN_VALID = False
RUN_TEST  = False
USE_DEVICE = 'GPU' #'CPU'  # 'GPU'

import random
import os, sys
import time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # Use standard DataLoader/Dataset

from tqdm import tqdm
from _cfg import cfg
from copy import deepcopy
from types import MethodType

import torch.nn.functional as F
import timm
from timm.models.convnext import ConvNeXtBlock

from monai.networks.blocks import UpSample, SubpixelUpsample
from types import SimpleNamespace
from timm.models.layers import trunc_normal_

import pandas as pd
import matplotlib.pyplot as plt
import collections
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

# Install einops if not present
try:
    from einops import rearrange
except ImportError:
    !pip install einops -q
    from einops import rearrange


try: 
    import monai
except: 
    !pip install --no-deps monai -q

data_paths_str = "./datasetfiles/FlatVel_A/data/*.npy"
label_paths_str = "./datasetfiles/FlatVel_A/model/*.npy"


cfg= SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.local_rank = 0
cfg.seed = 123
cfg.subsample = 100 #None

cfg.file_pairs = list(zip(sorted(glob.glob("./datasetfiles/FlatVel_A/data/*.npy")), sorted(glob.glob("./datasetfiles/FlatVel_A/model/*.npy"))))
# cfg.file_pairs = list(zip(data_paths, label_paths))
data_paths = sorted(glob.glob("./datasetfiles/FlatVel_A/data/*.npy"))
label_paths = sorted(glob.glob("./datasetfiles/FlatVel_A/model/*.npy"))
cfg.backbone = "convnext_small.fb_in22k_ft_in1k"
cfg.ema = True
cfg.ema_decay = 0.99

cfg.epochs = 4
cfg.batch_size = 8  # 16
cfg.batch_size_val = 8 # 16

cfg.early_stopping = {"patience": 3, "streak": 0}
cfg.logging_steps = 10

# --- New Configs for Transformer and AMP ---
cfg.num_transformer_layers = 1 # Number of transformer layers in the added module
cfg.transformer_dropout = 0.1
cfg.use_amp = True # Enable Automatic Mixed Precision
cfg.amp_dtype = 'bfloat16' # 'bfloat16' or 'float16' - bfloat16 recommended if supported



####################
## EMA + Ensemble ##
####################

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.99, device=None):
        super().__init__()
        # Create EMA model on CPU, then move to specified device
        self.module = deepcopy(model).cpu() # Ensure deepcopy is on CPU first
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)



    @torch.no_grad()
    def _update(self, model, update_fn):
         # Ensure model params are on the target device (might be on different device than EMA if using DDP)
         # We assume model is on cfg.device
         model_state_dict = model.state_dict()
         for ema_v, model_k in zip(self.module.state_dict().values(), model_state_dict.keys()):
              model_v = model_state_dict[model_k].to(device=self.device) # Move model param to EMA device
              ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models).eval()

    def forward(self, x):
        output = None

        for m in self.models:
            # Ensure models are on the correct device and handle AMP if needed
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16 if x.device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16, enabled=x.device.type == 'cuda'): # Assuming AMP config is applied upstream or model handles it internally
                logits = m(x)

            if output is None:
                output = logits
            else:
                output += logits

        output /= len(self.models)
        return output
    
#############
## Decoder ##
#############

class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding: int = 0,
        stride: int = 1,
        norm_layer: nn.Module = nn.Identity, # Expecting a class, not an instance
        act_layer: nn.Module = nn.ReLU,     # Expecting a class
    ):
        super().__init__()

        self.conv= nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False, # Usually False when using BatchNorm/InstanceNorm
        )
        # Instantiate norm_layer here
        self.norm = norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
        # Instantiate act_layer here
        self.act= act_layer(inplace=True) if act_layer != nn.Identity else nn.Identity() # Handle Identity activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SCSEModule2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.GELU(), # Use GELU consistent with model
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention2d(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None or name.lower() == "identity":
            self.attention = nn.Identity() # Use nn.Identity for clarity
        elif name.lower() == "scse":
            self.attention = SCSEModule2d(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock2d(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer: nn.Module = nn.Identity,
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        scale_factor: int = 2,
        act_layer: nn.Module = nn.GELU, # Use GELU consistent with model
    ):
        super().__init__()

        # Upsample block
        if upsample_mode == "pixelshuffle":
            # SubpixelUpsample handles channel adjustment internally
            self.upsample= SubpixelUpsample(
                spatial_dims= 2,
                in_channels= in_channels,
                scale_factor= scale_factor,
            )
            upsample_out_channels = in_channels // (scale_factor ** 2) # Calculate output channels
        else: # "deconv", "bicubic", "bilinear", "nearest", "avgpool", "nontrainable"

            upsample_out_channels = in_channels if upsample_mode != "deconv" else in_channels # Let's keep it simple for now, assume deconv might change channels or leave them same
            # Re-check MONAI UpSample source: it takes out_channels for deconv
            if upsample_mode == "deconv":
                self.upsample = UpSample(
                    spatial_dims= 2,
                    in_channels= in_channels,
                    scale_factor= scale_factor,
                    mode= upsample_mode,
                )
                upsample_out_channels = in_channels # Assuming deconv keeps channels same for now

            else: # nontrainable modes
                 self.upsample = UpSample(
                    spatial_dims= 2,
                    in_channels= in_channels,
                    out_channels= in_channels, # Non-trainable modes need explicit out_channels (same as in_channels for skip)
                    scale_factor= scale_factor,
                    mode= upsample_mode,
                 )
                 upsample_out_channels = in_channels

        if intermediate_conv:
            k= 3
            # Intermediate conv applies to the skip connection OR the upsampled features if skip is None
            # If skip is not None, it applies to skip_channels -> skip_channels
            # If skip is None, it applies to upsampled_out_channels -> upsampled_out_channels
            intermediate_in_channels = skip_channels if skip_channels != 0 else upsample_out_channels
            intermediate_out_channels = intermediate_in_channels # Usually keeps channels same
            self.intermediate_conv = nn.Sequential(
                ConvBnAct2d(intermediate_in_channels, intermediate_out_channels, k, k//2, norm_layer=norm_layer, act_layer=act_layer),
                ConvBnAct2d(intermediate_out_channels, intermediate_out_channels, k, k//2, norm_layer=norm_layer, act_layer=act_layer),
                )
        else:
            self.intermediate_conv= None

        # Input to the first conv after upsampling and concatenation is upsampled_out_channels + skip_channels
        conv1_in_channels = upsample_out_channels + skip_channels

        self.attention1 = Attention2d(
            name= attention_type,
            in_channels= conv1_in_channels,
            )

        self.conv1 = ConvBnAct2d(
            conv1_in_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
            act_layer= act_layer,
        )

        self.conv2 = ConvBnAct2d(
            out_channels,
            out_channels,
            kernel_size= 3,
            padding= 1,
            norm_layer= norm_layer,
            act_layer= act_layer,
        )
        self.attention2 = Attention2d(
            name= attention_type,
            in_channels= out_channels,
            )

    def forward(self, x, skip=None):
        # x is the input from the previous decoder block (or deepest encoder feature)
        # skip is the feature map from the corresponding encoder stage
        x = self.upsample(x)

        if self.intermediate_conv is not None:
            if skip is not None:
                # Apply intermediate conv to skip connection
                skip = self.intermediate_conv(skip)
            else:
                 pass # Do nothing if intermediate_conv exists but skip is None

        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                 # Center crop the larger one or pad the smaller one
                 # Let's center crop the skip connection if it's larger
                 target_h, target_w = x.shape[-2:]
                 skip_h, skip_w = skip.shape[-2:]
                 if skip_h > target_h or skip_w > target_w:
                     # print(f"Warning: Cropping skip connection from {skip.shape} to match {x.shape}")
                     # Calculate cropping amounts
                     crop_h = max(0, skip_h - target_h)
                     crop_w = max(0, skip_w - target_w)
                     # Apply center crop
                     skip = skip[:, :, crop_h // 2 : crop_h // 2 + target_h, crop_w // 2 : crop_w // 2 + target_w]
                 elif skip_h < target_h or skip_w < target_w:
                      # print(f"Warning: Padding skip connection from {skip.shape} to match {x.shape}")
                      # Calculate padding amounts
                      pad_h = max(0, target_h - skip_h)
                      pad_w = max(0, target_w - skip_w)
                      # Apply zero padding
                      skip = F.pad(skip, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class UnetDecoder2d(nn.Module):
    """
    Unet decoder based on timm features_only output order (deep to shallow).
    Source: https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self,
        encoder_channels: list[int], # Expects channels from deep to shallow [C_s3, C_s2, C_s1, C_s0]
        decoder_channels: list = (256, 128, 64, 32), # Deep to shallow decoder channels
        scale_factors: list = (2,2,2,2), # Upsample scale factor for each block
        norm_layer: nn.Module = nn.InstanceNorm2d, # Use InstanceNorm consistent with encoder replacement
        attention_type: str = None,
        intermediate_conv: bool = False,
        upsample_mode: str = "deconv",
        act_layer: nn.Module = nn.GELU, # Use GELU
    ):
        super().__init__()

        # Validate input channel counts
        num_encoder_stages = len(encoder_channels)
        num_decoder_blocks = len(decoder_channels)

        if num_decoder_blocks != num_encoder_stages:

             if num_decoder_blocks == num_encoder_stages - 1:
                 print(f"Warning: Number of decoder blocks ({num_decoder_blocks}) does not match encoder stages ({num_encoder_stages}). Adjusting decoder channels.")

                 in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
                 skip_channels_list = encoder_channels[1:] + [0] # C_s2, C_s1, C_s0, 0

                 actual_decoder_channels = list(decoder_channels)


             if num_decoder_blocks != num_encoder_stages:
                 print(f"Warning: Number of decoder block outputs ({num_decoder_blocks}) does not match number of encoder stages ({num_encoder_stages}). Adjusting number of decoder blocks to match encoder stages.")
                 if len(decoder_channels) != num_encoder_stages:
                     print(f"Adjusting decoder_channels length from {len(decoder_channels)} to {num_encoder_stages}.")
                     # This requires deciding how to adjust channels if the lengths don't match.
                     # For simplicity, let's require `len(decoder_channels) == len(encoder_channels)`.
                     raise ValueError(f"Number of decoder_channels ({len(decoder_channels)}) must match number of encoder stages ({num_encoder_stages}) for this decoder structure.")


             self.decoder_channels = decoder_channels # Store output channels of blocks

             # Input channels for blocks: C_s3 (for block 0), then DC0, DC1, DC2...
             in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])

             # Skip channels for blocks: C_s2, C_s1, C_s0, ... (last skip might be 0)
             # Skips are feats[1], feats[2], ..., feats[N-1]. Last block skip is 0.
             skip_channels_list = encoder_channels[1:] + [0]

             self.blocks = nn.ModuleList()
             for i in range(num_decoder_blocks):
                  self.blocks.append(
                      DecoderBlock2d(
                          in_channels= in_channels[i],
                          skip_channels= skip_channels_list[i], # Corresponding skip channel
                          out_channels= decoder_channels[i], # Output channel of this block
                          norm_layer= norm_layer,
                          attention_type= attention_type,
                          intermediate_conv= intermediate_conv,
                          upsample_mode= upsample_mode,
                          scale_factor= scale_factors[i],
                      )
                  )


    def forward(self, deep_feature: torch.Tensor, skip_features: list[torch.Tensor]):
        x = deep_feature
        decoder_outputs = [x] # Store intermediate decoder block outputs (optional, but maybe useful)

        # Loop through decoder blocks
        for i, block in enumerate(self.blocks):
            # The skip connection for block i comes from skip_features[i]
            skip = skip_features[i] if i < len(skip_features) else None # Handle case where last block has no skip
            x = block(x, skip=skip)
            decoder_outputs.append(x)
        return decoder_outputs # [input_s3, output_block0, output_block1, ...]


class SegmentationHead2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: tuple[int] | int = 1, # Use tuple or int for clarity
        kernel_size: int = 3,
        mode: str = "nontrainable",
        norm_layer: nn.Module = nn.Identity, # Add norm/act options consistent with blocks
        act_layer: nn.Module = nn.Identity,
    ):
        super().__init__()
        self.conv= ConvBnAct2d( # Use the standard ConvBnAct2d block
            in_channels, out_channels, kernel_size= kernel_size,
            padding= kernel_size//2, norm_layer=norm_layer, act_layer=act_layer,
        )
        # Use a tuple scale_factor even if it's 1,1 for 2D
        if isinstance(scale_factor, int):
             scale_factor = (scale_factor, scale_factor)

        # Only add upsample if scale_factor > 1
        if scale_factor[0] > 1 or scale_factor[1] > 1:
             self.upsample = UpSample(
                 spatial_dims= 2,
                 in_channels= out_channels, # Upsample operates on the output of the conv
                 out_channels= out_channels,
                 scale_factor= scale_factor,
                 mode= mode,
             )
        else:
             self.upsample = nn.Identity() # No upsampling needed


    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x) # Identity if scale_factor is 1
        return x


#############
## Encoder ##
#############


class TemporalSpatialTransformer(nn.Module):
    """
    Applies transformer attention along temporal and spatial dimensions
    of a 4D feature map (B, C, H, W).
    """
    def __init__(self, embed_dim: int, seq_len_h: int, seq_len_w: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len_h = seq_len_h
        self.seq_len_w = seq_len_w
        self.num_layers = num_layers

        # Learnable 1D positional embeddings for Height and Width dimensions
        self.pos_embed_h = nn.Parameter(torch.randn(1, 1, seq_len_h, 1, embed_dim)) # Shape (1, 1, H, 1, E)
        self.pos_embed_w = nn.Parameter(torch.randn(1, 1, 1, seq_len_w, embed_dim)) # Shape (1, 1, 1, W, E)
        # Initialize Positional Embeddings
        trunc_normal_(self.pos_embed_h, std=.02)
        trunc_normal_(self.pos_embed_w, std=.02)


        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8, # Typical number of heads, could make configurable
            dim_feedforward=embed_dim * 4, # Typical feedforward dim
            dropout=dropout,
            activation="gelu", # Use GELU
            batch_first=False, # Transformer expects (Sequence, Batch, Embed)
            norm_first=True, # Pre-LayerNorm is common in recent transformers
        )
        self.temporal_transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.spatial_transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers) # Share layers or separate? Separate for clarity.
        # If sharing, need to be careful about state. Let's make copies of the layers.
        transformer_layers_h = [deepcopy(transformer_layer) for _ in range(num_layers)]
        transformer_layers_w = [deepcopy(transformer_layer) for _ in range(num_layers)]
        self.temporal_transformer = nn.TransformerEncoder(nn.ModuleList(transformer_layers_h), num_layers=num_layers)
        self.spatial_transformer = nn.TransformerEncoder(nn.ModuleList(transformer_layers_w), num_layers=num_layers)


    def forward(self, x):
        # Input shape: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.embed_dim, f"Input channel mismatch: {C} != {self.embed_dim}"

        if H != self.seq_len_h or W != self.seq_len_w:
             # print(f"Warning: Feature map spatial size ({H},{W}) does not match expected ({self.seq_len_h},{self.seq_len_w}) for Transformer. Padding/Cropping.")
             # Calculate padding/cropping
             pad_h = max(0, self.seq_len_h - H)
             pad_w = max(0, self.seq_len_w - W)
             crop_h = max(0, H - self.seq_len_h)
             crop_w = max(0, W - self.seq_len_w)

             if pad_h > 0 or pad_w > 0:
                 x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
             if crop_h > 0 or crop_w > 0:
                  x = x[:, :, crop_h // 2 : crop_h // 2 + self.seq_len_h, crop_w // 2 : crop_w // 2 + self.seq_len_w]

             # Update H, W after padding/cropping
             B, C, H, W = x.shape
             assert H == self.seq_len_h and W == self.seq_len_w, "Padding/cropping failed to match dimensions."


        identity = x # Residual connection

        pos_embed_h_4d = self.pos_embed_h.squeeze(dim=3).permute(0, 4, 2, 1) # (1, E, H, 1)
        pos_embed_w_4d = self.pos_embed_w.squeeze(dim=2).permute(0, 4, 1, 2) # (1, E, 1, W)


        # --- Temporal Attention ---
        # Input (B, C, H, W). Treat each (C, W) slice as a batch element, sequence length H.
        # Reshape to (H, B*W, C) for Transformer
        x_temp = x + pos_embed_h_4d.to(x.dtype) # Add PE for H dimension (broadcasts over B, W)
        x_temp = x_temp.permute(2, 0, 3, 1).reshape(H, B * W, C) # (H, B*W, C)
        x_temp = self.temporal_transformer(x_temp) # (H, B*W, C)
        # Reshape back to (B, C, H, W)
        x_temp = x_temp.reshape(H, B, W, C).permute(1, 3, 0, 2) # (B, C, H, W)
        x = x_temp


        # --- Spatial (Geophone) Attention ---
        # Input (B, C, H, W). Treat each (C, H) slice as a batch element, sequence length W.
        # Reshape to (W, B*H, C) for Transformer
        x_spat = x + pos_embed_w_4d.to(x.dtype) # Add PE for W dimension (broadcasts over B, H)
        x_spat = x_spat.permute(3, 0, 2, 1).reshape(W, B * H, C) # (W, B*H, C)
        x_spat = self.spatial_transformer(x_spat) # (W, B*H, C)
        # Reshape back to (B, C, H, W)
        x_spat = x_spat.reshape(W, B, H, C).permute(1, 3, 2, 0) # (B, C, H, W)
        x = x_spat

        # Add residual connection
        x = x + identity

        return x


class Net(nn.Module):
    def __init__(
        self,
        cfg,
        backbone: str,
        pretrained: bool = True,
    ):
        super().__init__()

        self.cfg = cfg

        # Encoder - use features_only=True to get multi-scale features
        # We will manually modify the stem after loading the model.
        self.backbone= timm.create_model(
            backbone,
            in_chans= 5,
            pretrained= pretrained,
            features_only= True, # Get list of features from stages
            drop_path_rate=0.0,
        )

        encoder_feature_info = self.backbone.feature_info
        encoder_channels = [info['num_chs'] for info in encoder_feature_info] # [C_s3, C_s2, C_s1, C_s0] for a 4-stage model

        self._update_stem(backbone)

        H_last_feat = max(1, round(71 / (2**3))) # Approx H after 3 stages of 2x downsampling
        W_last_feat = max(1, round(72 / (2**3))) # Approx W after 3 stages of 2x downsampling
        C_last_feat = encoder_channels[0] # Channel count of the deepest feature (from feature_info)

        # Instantiate the Transformer module
        self.transformer_module = TemporalSpatialTransformer(
             embed_dim=C_last_feat,
             seq_len_h=H_last_feat,
             seq_len_w=W_last_feat,
             num_layers=getattr(cfg, 'num_transformer_layers', 1),
             dropout=getattr(cfg, 'transformer_dropout', 0.1),
        )

        # Decoder - expects encoder channels deep to shallow [C_s3, C_s2, C_s1, C_s0]
        # Needs to match the number of decoder blocks to the number of encoder stages/features
        # Let's use default decoder_channels or make it configurable.
        # Default: (256, 128, 64, 32)
        default_decoder_channels = (256, 128, 64, 32)
        # Ensure number of decoder blocks matches number of encoder features
        if len(default_decoder_channels) != len(encoder_channels):
             print(f"Warning: Adjusting number of decoder channels from {len(default_decoder_channels)} to match encoder stages {len(encoder_channels)}")
             # Simple strategy: Take the first N channels or repeat the last one
             num_stages = len(encoder_channels)
             if len(default_decoder_channels) >= num_stages:
                 decoder_channels = default_decoder_channels[:num_stages]
             else:
                 # Repeat the last channel or interpolate? Simple repeat:
                 decoder_channels = list(default_decoder_channels) + [default_decoder_channels[-1]] * (num_stages - len(default_decoder_channels))
        else:
             decoder_channels = default_decoder_channels

        # Use InstanceNorm and GELU in the decoder blocks consistent with the encoder replacement
        self.decoder= UnetDecoder2d(
            encoder_channels= encoder_channels, # [C_s3, C_s2, C_s1, C_s0]
            decoder_channels= list(decoder_channels), # [DC0, DC1, DC2, DC3] - explicit list
            norm_layer= nn.InstanceNorm2d, # Use InstanceNorm
            act_layer= nn.GELU, # Use GELU
            attention_type= 'scse', # Keep SCSE attention if desired
        )

        # Segmentation Head - expects the output of the last decoder block
        self.seg_head= SegmentationHead2d(
            in_channels= decoder_channels[-1], # Output channel of the last decoder block (DC3)
            out_channels= 1,
            scale_factor= 1, # Output is already 70x70, no need to scale up further
            norm_layer= nn.Identity, # No norm/act in head usually
            act_layer= nn.Identity,
        )

        # Apply model modifications (activations, norms)
        # Make sure LayerNorm within Transformer is NOT replaced by InstanceNorm
        self.replace_activations(self.backbone, log=True)
        self.replace_norms(self.backbone, log=True)
        # No need to replace forwards if we handle the stem manually and use features_only=True output.
        # self.replace_forwards(self.backbone, log=True) # REMOVE THIS CALL


    def _update_stem(self, backbone):
        """
        Modifies the initial layers (stem) of the ConvNeXt backbone
        to handle the (C, T, W) -> (5, 1000, 70) input shape.
        """
        if not hasattr(backbone, 'stem'):
             print(f"Warning: Backbone {backbone.__class__.__name__} does not have a 'stem' attribute. Stem modification skipped.")
             return

        # Assuming the original stem is nn.Sequential(Conv2d, LayerNorm2d)
        original_stem_conv = backbone.stem[0]
        original_stem_norm = backbone.stem[1]


        # Get original conv and norm parameters
        in_chans = original_stem_conv.in_channels
        out_chans = original_stem_conv.out_channels # C_stem

        # Define the new stem sequential block
        new_stem_layers = []

        # Add padding used in original code
        # The original code used ReflectionPad2d((1,1,80,80))
        # This pads Width by 1 on each side (70->72) and Height by 80 on each side (1000->1160)
        new_stem_layers.append(nn.ReflectionPad2d((1, 1, 80, 80)))

        # Add the original Conv2d layer (potentially with modified stride/padding)
        # Original code modified stride to (4, 1) and padding to (0, 2) on the original conv.
        original_kernel = original_stem_conv.kernel_size
        original_padding = original_stem_conv.padding
        original_stride = original_stem_conv.stride

        # Let's make sure the stride/padding are set *before* adding it to the sequence
        stem_conv1 = nn.Conv2d(
             in_channels=in_chans,
             out_channels=out_chans,
             kernel_size=original_kernel,
             stride=(4, 1), # Modified stride from original code
             padding=(0, 2), # Modified padding from original code
             bias=original_stem_conv.bias is not None # Keep bias state
        )
        with torch.no_grad():
             stem_conv1.weight.copy_(original_stem_conv.weight)
             if original_stem_conv.bias is not None:
                  stem_conv1.bias.copy_(original_stem_conv.bias)

        new_stem_layers.append(stem_conv1)

        # Add the second custom conv layer from the original code
        # kernel=(4, 4), stride=(4, 1), padding=(0, 1)
        # Input channels = output channels of stem_conv1 (out_chans)
        # Output channels = out_chans (seems to keep channels the same)
        # This second conv further downsamples Height by 4x.
        stem_conv2 = nn.Conv2d(
             in_channels=out_chans,
             out_channels=out_chans, # Assumed based on original code's new_conv initialization
             kernel_size=(4, 4),
             stride=(4, 1),
             padding=(0, 1),
             bias=False # Assumed bias=False for new conv before norm/activation
        )
        # Initialize stem_conv2 weights/bias if needed, original code copied/repeated weights
        # Let's use default Kaiming init or similar, copying repeated weights is unusual.
        nn.init.kaiming_normal_(stem_conv2.weight, mode='fan_out', nonlinearity='relu')
        # The original code copied original stem weights *repeatedly*. This is highly specific and possibly incorrect.
        # Let's use standard initialization for the new conv layer.
        # If bias exists, initialize to zero.
        if stem_conv2.bias is not None:
             nn.init.constant_(stem_conv2.bias, 0)

        new_stem_layers.append(stem_conv2)


        stem_norm = nn.LayerNorm(out_chans, eps=1e-6) # ConvNeXt uses eps=1e-6
        new_stem_layers.append(stem_norm)


        if hasattr(backbone, 'stem') and isinstance(backbone.stem, nn.Sequential):
             backbone.stem = nn.Sequential(*new_stem_layers)
             print(f"Replaced backbone stem with custom sequential block.")

        else:
             print(f"Warning: Backbone stem structure is not as expected for modification.")


    def replace_activations(self, module, log=False):
        """ Recursively replaces specific activation functions with GELU. """
        if log:
            # print(f"Replacing activations with GELU...")
            pass

        for name, child in module.named_children():
            if isinstance(child, (
                nn.ReLU, nn.LeakyReLU, nn.Mish, nn.Sigmoid,
                nn.Tanh, nn.Softmax, nn.Hardtanh, nn.ELU,
                nn.SELU, nn.PReLU, nn.CELU, nn.SiLU,
            )):
                # Replace activation instance
                if log: print(f"  Replacing {type(child).__name__} at {name} with GELU")
                setattr(module, name, nn.GELU())
            else:
                # Recurse into child modules
                self.replace_activations(child, log=log)

    def replace_norms(self, mod, log=False):
        """ Recursively replaces specific normalization layers with InstanceNorm2d, skipping LayerNorm. """
        if log:
            # print(f"Replacing norms with InstanceNorm2d (skipping LayerNorm)...")
            pass

        for name, c in mod.named_children():
            # Skip LayerNorm as it's used in Transformers and ConvNeXt often uses it after stem
            if isinstance(c, nn.LayerNorm):
                # if log: print(f"  Skipping LayerNorm at {name}")
                continue

            # Get feature size (handle different norm types)
            n_feats= None
            if isinstance(c, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                n_feats= c.num_features
            elif isinstance(c, (nn.GroupNorm,)):
                n_feats= c.num_channels
            # nn.LayerNorm is already skipped above

            if n_feats is not None:
                # Create new InstanceNorm2d layer
                # if log: print(f"  Replacing {type(c).__name__} at {name} with InstanceNorm2d")
                new = nn.InstanceNorm2d(
                    n_feats,
                    affine=True, # Keep affine=True to match BatchNorm/LayerNorm behavior
                    track_running_stats=False, # InstanceNorm typically doesn't track stats
                )
                setattr(mod, name, new)
            else:
                # Recurse into child modules
                self.replace_norms(c, log=log)

    # Removed replace_forwards as it's not needed with features_only=True approach


    def forward(self, batch):
        x = batch # Input (B, 5, 1000, 70)
        B_ = x.shape[0]

        # AMP context for mixed precision
        # Use bfloat16 if available and requested, otherwise float16
        amp_dtype = None
        if hasattr(cfg, 'use_amp') and cfg.use_amp and cfg.device.type == 'cuda':
             amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and getattr(cfg, 'amp_dtype', 'bfloat16') == 'bfloat16' else torch.float16
             if amp_dtype == torch.float16:
                 print("Warning: Using float16 AMP without GradScaler might cause issues during training.")
             # Enable AMP only if dtype was successfully determined
             amp_enabled = (amp_dtype is not None)
        else:
             amp_enabled = False
             amp_dtype = None # Explicitly None if AMP is off


        # if not hasattr(self.backbone, 'stem') or not hasattr(self.backbone, 'stages') or not hasattr(self.backbone, 'downsample_layers'):
        #     raise AttributeError("Backbone structure not as expected for manual forward pass.")

        with torch.autocast(device_type=self.cfg.device.type, dtype=amp_dtype, enabled=amp_enabled):
             x_in = x # Keep original input for inference-time flip

             # Pass through the custom stem
             # The stem expects (B, C, H, W) where C=5, H=1000, W=70
            #  stem_out = self.backbone.stem(x) # Output (B, C_stem, H_stem, W_stem) e.g. (B, 128, 71, 72)
             x = self.backbone.stem_0(x) #pass through the first layer (Conv2d)
             stem_out = self.backbone.stem_1(x) # Pass through the second layer (LayerNorm2d) # output (B, C, H, W) 

             # Pass through subsequent stages and downsamples
             # Collect intermediate features for the decoder
             features_for_decoder = []
             # Stage 0 is the first stage after the stem. Add its output.
            #  x = self.backbone.stages[0](stem_out)
             x = self.backbone.stages_0(stem_out)
             features_for_decoder.append(x) # s0

             # Pass through subsequent stages and downsamples
             # Stages 1, 2, 3 are connected via downsample layers
             # The i-th downsample_layer is applied before the i+1-th stage
             for i in range(len(self.backbone.downsample_layers)): # Loop through downsample layers
            #  for i in range(len(self.backbone.downsample)): # Loop through downsample layers
                 # downsample_layers[i] is applied to the output of stages[i]
                #  x = self.backbone.downsample[i](x)
                 x = self.backbone.downsample_layers[i](x)
                 # The output of downsample_layers[i] is input to stages[i+1]
                 x = self.backbone.stages[i+1](x)
                 features_for_decoder.append(x) # s1, s2, s3...

             # features_for_decoder is now [s0, s1, s2, s3] (shallow to deep)
             # Reverse for decoder input [s3, s2, s1, s0]
             feats = features_for_decoder[::-1]

             # Apply Transformer module to the LAST feature map (s3)
             last_feat_map = feats[0] # This is s3

             # Ensure the transformer input spatial dimensions match the expected seq_len_h/w
             # The Transformer module handles padding/cropping internally based on its init values.
             trans_feat = self.transformer_module(last_feat_map)

             # Prepare features for the decoder: [transformed_s3, s2, s1, s0]
             decoder_input_feats = [trans_feat] + feats[1:]

             # Decoder forward - pass the main input (transformed s3) and the list of skips [s2, s1, s0]
             decoder_outs = self.decoder(decoder_input_feats[0], decoder_input_feats[1:])

             # Segmentation Head takes the output of the last decoder block
             seg_head_input = decoder_outs[-1] # This is the output of the shallowest decoder block

             # Segmentation Head
             x_seg = self.seg_head(seg_head_input)


             output_h, output_w = x_seg.shape[-2:]
             target_h, target_w = 70, 70
             if output_h != target_h or output_w != target_w:
                 # Calculate cropping amounts
                 crop_h = max(0, output_h - target_h)
                 crop_w = max(0, output_w - target_w)
                 # Apply center crop
                 x_seg = x_seg[:, :, crop_h // 2 : crop_h // 2 + target_h, crop_w // 2 : crop_w // 2 + target_w]
                 # print(f"Cropped head output from {output_h}x{output_w} to {x_seg.shape[-2:]}")


             x_seg = x_seg * 1500 + 3000 # Apply scaling

        # Inference-time flip augmentation
        if not self.training:
             # Apply spatial flip to the input data
             x_in_flipped = torch.flip(x_in, dims=[-1]) # Flip geophone dimension
             # Process the flipped input through the same model forward path
             # This recursive call will also apply AMP internally
             flipped_output = self.forward(x_in_flipped)
             # Flip the output spatially back
             flipped_output_spatial_flipped = torch.flip(flipped_output, dims=[-1]) # Flip map width dimension

             # Average the original and flipped outputs
             x_seg = torch.mean(torch.stack([x_seg, flipped_output_spatial_flipped]), dim=0)


        return x_seg
    

def set_seed(seed=cfg.seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(cfg.seed)
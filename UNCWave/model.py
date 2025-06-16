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


class CustomDataset(torch.utils.data.Dataset):
    # --- Rewriting CustomDataset to return a time window ---
    # This assumes each file is long enough (>= 1000 time steps).
    # If files are shorter, padding or different sampling strategy is needed.
    # Let's assume files are long enough for now.
    # The index_map should store (file_idx, start_time_idx_of_window).

    def __init__(
        self,
        cfg,
        file_pairs,
        mode = "train",
    ):
        self.cfg = cfg
        self.mode = mode
        self.file_pairs = file_pairs

        self.data, self.labels = self._load_data_arrays()
        print(f"[CustomDataset - init ]:self.data shape: {self.data[0].shape} | self.labels shape {self.labels[0].shape}")

        if not self.data:
             self.total_samples = 0
             self.index_map = []
             print(f"Dataset '{self.mode}' created with 0 total samples (no data loaded).")
             return # Exit init if no data

        self.samples_per_file = self.data[0].shape[0] # Total time steps in a file
        self.index_map = [] # Number of geophones
        print(f"[CustomDataset - init ]:samples_per_file : {self.samples_per_file}")  #500

        # Build list of (file_idx, window_start_idx) pairs
        for file_idx, file_data in enumerate(self.data):
            print(f"[CustomDataset - init ]:file_idx : {file_idx} ") 
            for b_idx in range(self.samples_per_file):
                self.index_map.append((file_idx, b_idx))
        
        self.total_samples = len(self.index_map)
        print(f"[CustomDataset - init ]:total_samples : {self.total_samples}")  #500
        print(f"[CustomDataset - init ]:index_map : {self.index_map}")  #500

    def _load_data_arrays(self, ):

        data_arrays = []
        label_arrays = []
        mmap_mode = "r" # Use read-only memory map

        # Only load a subset if subsample is much smaller than total possible samples
        # This avoids memory mapping huge amounts if only a few samples are needed.
        # However, the current logic always memory maps all files listed in file_pairs.
        # For simplicity, we'll keep mmap_mode="r" as it's memory efficient for large files.

        for data_fpath, label_fpath in tqdm(
                        self.file_pairs, desc=f"Loading {self.mode} data (mmap)",
                        disable=self.cfg.local_rank != 0 or not self.file_pairs):
            try:
                # Load the numpy arrays using memory mapping
                arr = np.load(data_fpath, mmap_mode=mmap_mode)
                lbl = np.load(label_fpath, mmap_mode=mmap_mode)
                print(f"[CustomDataset - load data arrays ]:arr shape: {arr.shape} | lbl shape: {lbl.shape}")
                lbl = np.squeeze(lbl, axis=1)
                print(f"[CustomDataset - load data arrays ]:lbl after squeeze shape: {lbl.shape}")
                # print(f"Loaded {data_fpath}: {arr.shape}, {lbl.shape}") # Too verbose
                data_arrays.append(arr)
                label_arrays.append(lbl)
            except FileNotFoundError:
                print(f"Error: File not found - {data_fpath} or {label_fpath}", file=sys.stderr)
            except Exception as e:
                print(f"Error loading file pair: {data_fpath}, {label_fpath}", file=sys.stderr)
                print(f"Error: {e}", file=sys.stderr)
                continue

        if self.cfg.local_rank == 0 and self.file_pairs:
            print(f"Finished loading {len(data_arrays)} file pairs for {self.mode} mode.")

        return data_arrays, label_arrays
    
    def __getitem__(self, idx):
        file_idx, start_time_idx = self.index_map[idx]
        print(f"[CustomDataset - getitem ]:file_idx {file_idx} | file_idx{start_time_idx}")

        # Access the data window and corresponding label (assuming label is the same map for all time steps in a file)
        # If the label changes per time step, the label loading/indexing needs adjustment.
        # The requested output is (B, 1, 70, 70), suggesting a single map per input window.
        # Let's assume the geological map (label) is constant for all time steps within a single data file.
        # So we load the label only once per file and return the same label for any window from that file.

        x_sample = self.data[file_idx][start_time_idx] # (Samples_per_file, Channels, Geophones) -> (S, C, W) = (S, 5, 70)
        y_sample = self.labels[file_idx][start_time_idx] # (Samples_per_file, Map_Height, Map_Width) -> (S, H', W') = (S, 70, 70)
        print(f"[CustomDataset - getitem ]:x_full_file shape {x_sample.shape} | y_full_file shape: {y_sample.shape}")
        # y_full_file = np.squeeze(y_full_file, axis=1)
        # print(f"[CustomDataset - getitem ]:np.squeeze(y_full_file, axis=1)-shape: {y_full_file.shape}")

        # --- Augmentations (apply to window and label) ---
        x_augmented = x_sample.copy()
        y_augmented = y_sample.copy()
       
        if self.mode == "train":
            # Temporal flip (e.g., flipping across time dimension - dim 1)
            if np.random.random() < 0.5:
                x_augmented = x_augmented[:, ::-1, :].copy()  # Flip Time (dim 1) and copy

            # Spatial flip (geophones and map width)
            if np.random.random() < 0.5:
                x_augmented = x_augmented[:, :, ::-1].copy()  # Flip Geophones (dim 2)
                y_augmented = y_augmented[:, ::-1].copy()     # Flip Map Width (dim 1)

        print(f"[CustomDataset - getitem ]:x_augmented shape {x_augmented.shape} | y_augmented shape: {y_augmented.shape}")
        # Convert numpy to torch tensors
        # x_tensor needs to be (Channels, Time, Geophones) -> (5, 1000, 70)
        x_tensor = torch.from_numpy(x_augmented).float() #.permute(1, 0, 2) # From (1000, 5, 70) to (5, 1000, 70)
        # y_tensor needs to be (1, Map_Height, Map_Width) -> (1, 70, 70)
        y_tensor = torch.from_numpy(y_augmented).float().unsqueeze(0) # From (70, 70) to (1, 70, 70)
        print(f"[CustomDataset - getitem ]:x_tensor shape {x_tensor.shape} | y_tensor shape: {y_tensor.shape}")

        return x_tensor, y_tensor

    def __len__(self, ):
        print("[CustomDataset - __len__]: self.total_samples: ", self.total_samples)
        return self.total_samples
    

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
            # Move buffers like running_mean/var for BatchNorm if they exist (though we replace norms)
            # This loop handles moving state_dict items
            # print(f"Moving EMA model to device: {device}")
            # for k, v in self.module.state_dict().items():
            #     print(f"  Moving {k} to {device}")
            #     self.module.state_dict()[k].copy_(v.to(device)) # This copy might be slow


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
             # For "deconv", need to set out_channels explicitly if different from in_channels
             # For others, out_channels is same as in_channels
             # Assuming deconv is primary mode and might change channels
            upsample_out_channels = in_channels if upsample_mode != "deconv" else in_channels # Let's keep it simple for now, assume deconv might change channels or leave them same
            # Re-check MONAI UpSample source: it takes out_channels for deconv
            if upsample_mode == "deconv":
                 # Deconv output channels usually match input channels for skip connection concatenation
                 # Let's make deconv out_channels explicit: in_channels -> out_channels
                 # But wait, the total input to conv1 is upsampled_channels + skip_channels.
                 # If upsampling changes channels, this needs to be handled.
                 # Standard practice is upsampling brings feature map to spatial size of skip, channels might match input to block or be related to skip.
                 # Let's assume for 'deconv', MONAI's UpSample can handle channel adjustment or keeps it `in_channels`.
                 # Re-reading MONAI UpSample: `out_channels` is only used for "nontrainable" upsample. For "deconv", it infers.
                 # Okay, let's assume upsample output channels are `in_channels` for deconv too, before concatenation.
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
                 # Should not happen in a standard U-Net skip connection flow, but handle if used differently
                 # If skip is None, intermediate conv might apply to upsampled feature map
                 # The original code's intermediate conv logic is a bit ambiguous here.
                 # Let's stick to the common case: intermediate_conv applies to skip.
                 pass # Do nothing if intermediate_conv exists but skip is None

        if skip is not None:
            # Ensure spatial sizes match before concatenation
            # print(f"Upsampled shape: {x.shape}, Skip shape: {skip.shape}")
            # If shapes don't match exactly (due to padding/strides), need cropping/padding
            # Assuming timm/monai handle this correctly or input sizes are compatible
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
             # The first encoder channel is the input to the first decoder block
             # The rest are skips. So we need num_decoder_blocks = num_encoder_stages.
             # If encoder_channels = [C_s3, C_s2, C_s1, C_s0] (4 stages)
             # Decoder needs 4 blocks: s3->skip(s2), s2_dec->skip(s1), s1_dec->skip(s0), s0_dec->skip(None or final).
             # The original code sets decoder_channels to (256, 128, 64, 32).
             # If len(encoder_channels) == 4, it uses decoder_channels[1:] = (128, 64, 32). This implies 3 decoder blocks?
             # This seems mismatched. Let's align them: num_decoder_blocks = num_encoder_stages.
             # And the decoder_channels should specify the output channels of each decoder block (deep to shallow).
             # Let's correct the logic based on a standard U-Net structure.
             # Input to Decoder: [s_N, s_{N-1}, ..., s_0] where s_N is deepest.
             # Block 1 input: s_N, skip s_{N-1}, output d_{N-1}
             # Block 2 input: d_{N-1}, skip s_{N-2}, output d_{N-2}
             # ...
             # Block N input: d_1, skip s_0, output d_0
             # Decoder output channels: [d_{N-1}, d_{N-2}, ..., d_0] (deepest block output -> shallowest block output)
             # The `decoder_channels` parameter should be the list of *output* channels for each block (deep to shallow).
             # If `decoder_channels` = (256, 128, 64, 32), these are the output channels of the 4 blocks.
             # Block 0 (deepest): Input C_s3, Skip C_s2, Output 256
             # Block 1: Input 256, Skip C_s1, Output 128
             # Block 2: Input 128, Skip C_s0, Output 64
             # Block 3: Input 64, Skip 0, Output 32 (assuming 4 encoder stages, the last block might not have a skip or use a placeholder)

             if num_decoder_blocks == num_encoder_stages - 1:
                 # Special case: if decoder_channels is one shorter, it might skip the first (deepest) block output channel?
                 # No, the original code has `if len(encoder_channels) == 4: decoder_channels= decoder_channels[1:]`
                 # This suggests if there are 4 encoder features, the decoder *channels* list is truncated.
                 # If ecs = [C_s3, C_s2, C_s1, C_s0], len=4. decoder_channels becomes (128, 64, 32), len=3.
                 # This would mean 3 decoder blocks. Where does s3 go? It would be the input to the first block.
                 # Block 0: Input C_s3, Skip C_s2, Output 128
                 # Block 1: Input 128, Skip C_s1, Output 64
                 # Block 2: Input 64, Skip C_s0, Output 32
                 # This seems plausible for a 4-stage encoder outputting 4 features.
                 print(f"Warning: Number of decoder blocks ({num_decoder_blocks}) does not match encoder stages ({num_encoder_stages}). Adjusting decoder channels.")
                 # Let's follow the original code's implicit intent: if 4 encoder stages, use 3 decoder blocks with channels (128, 64, 32)
                 # The input to the first decoder block is the output of the deepest encoder stage (feats[0] / s3).
                 # The skips are feats[1], feats[2], feats[3] (s2, s1, s0).

                 # Input channels for decoder blocks:
                 # Block 0: encoder_channels[0] (C_s3)
                 # Block 1: decoder_channels[0] (128)
                 # Block 2: decoder_channels[1] (64)
                 # Block 3 (if exists): decoder_channels[2] (32)
                 in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])

                 # Skip channels for decoder blocks:
                 # Block 0: encoder_channels[1] (C_s2)
                 # Block 1: encoder_channels[2] (C_s1)
                 # Block 2: encoder_channels[3] (C_s0)
                 # Block 3: 0 (no skip)
                 skip_channels_list = encoder_channels[1:] + [0] # C_s2, C_s1, C_s0, 0

                 # The number of blocks is determined by the length of decoder_channels list
                 actual_decoder_channels = list(decoder_channels)
                 # If len(encoder_channels) == 4 and original decoder_channels length is 4
                 # The original code does `decoder_channels = decoder_channels[1:]` -> len becomes 3
                 # This implies 3 decoder blocks (128, 64, 32) for 4 encoder stages [s3, s2, s1, s0].
                 # Block 0: input C_s3, skip C_s2, output 128
                 # Block 1: input 128, skip C_s1, output 64
                 # Block 2: input 64, skip C_s0, output 32
                 # This is a bit unconventional. Let's make the number of decoder channels *explicitly* define the blocks.
                 # If encoder has N stages, timm gives N features. Decoder should have N blocks.
                 # Let's define decoder_channels as a list of output channels for N blocks.
                 # If `encoder_channels = [C_s3, C_s2, C_s1, C_s0]` (len 4)
                 # and `decoder_channels_out = (256, 128, 64, 32)` (len 4)
                 # Block 0: input C_s3, skip C_s2, output 256
                 # Block 1: input 256, skip C_s1, output 128
                 # Block 2: input 128, skip C_s0, output 64
                 # Block 3: input 64, skip 0, output 32
                 # This makes more sense. Let's redefine based on this standard structure.

             # Redefined structure:
             # encoder_channels: [C_s3, C_s2, C_s1, C_s0] (deep to shallow)
             # decoder_channels_out: [DC0, DC1, DC2, DC3] (deep to shallow block outputs)
             # Number of blocks = len(encoder_channels) = len(decoder_channels_out)
             # Block 0: input encoder_channels[0] (C_s3), skip encoder_channels[1] (C_s2), output decoder_channels_out[0] (DC0)
             # Block i: input decoder_channels_out[i-1] (DC_{i-1}), skip encoder_channels[i+1] (C_{s_{N-(i+1)}}), output decoder_channels_out[i] (DC_i)
             # Last Block (N-1): input decoder_channels_out[N-2] (DC_{N-2}), skip encoder_channels[N] (C_s0), output decoder_channels_out[N-1] (DC_{N-1})
             # The skip for the *last* block (corresponding to the shallowest encoder feature) is the 2nd to last element in the encoder_channels list
             # Example: enc=[s3, s2, s1, s0], dec_out=[d0, d1, d2, d3]
             # Block 0: in s3, skip s2, out d0
             # Block 1: in d0, skip s1, out d1
             # Block 2: in d1, skip s0, out d2
             # Block 3: in d2, skip NONE, out d3 -> This structure doesn't match typical U-Net where last block connects to s0.
             # A typical U-Net decoder block takes the *previous decoder output* and the *corresponding encoder skip*.
             # Input to decoder block i: output of block i-1. Skip for block i: feature from encoder stage i.
             # If encoder features are [s0, s1, s2, s3] (shallow to deep)
             # Block 0 (deepest): input s3, skip s2, output d2
             # Block 1: input d2, skip s1, output d1
             # Block 2: input d1, skip s0, output d0
             # This requires encoder features in *shallow to deep* order for skips.

             # Let's reconcile with the original code's `ecs= [_["num_chs"] for _ in self.backbone.feature_info][::-1]`
             # This means `ecs` is [C_s3, C_s2, C_s1, C_s0].
             # The `forward` takes `feats` and does `res= [feats[0]]`, then loops using `feats[i]` as skip.
             # This implies `feats` is also [s3, s2, s1, s0].
             # `res=[s3]`
             # loop i=0: `skip=feats[0]=s3`. `b(s3, s3)`. WRONG.
             # loop i=1: `skip=feats[1]=s2`. `b(output_block0, s2)`. This is the correct skip connection logic.

             # Correct UnetDecoder2d forward logic matching original code's loop structure:
             # Takes `feats: list[torch.Tensor]` which is [s3, s2, s1, s0] (deep to shallow).
             # `x = feats[0]` (s3) # Input to the first decoder block
             # `res = [x]` # Store the initial input (s3) - maybe remove this?
             # Loop through decoder blocks i=0 to num_blocks-1
             # Block i takes input from block i-1's output (or s3 for block 0)
             # Block i takes skip from encoder stage i+1 in the `feats` list
             # (feats[1] for block 0, feats[2] for block 1, etc.)

             # Let's simplify the decoder structure in `Net` and Decoder `forward`:
             # Net:
             # `feats = self.backbone(x)` # [s3, s2, s1, s0]
             # `trans_s3 = self.transformer_module(feats[0])` # Process s3
             # `decoder_input = trans_s3` # Deepest feature is input to first decoder block
             # `decoder_skips = feats[1:]` # s2, s1, s0 are the skips
             # `decoder_out_list = self.decoder(decoder_input, decoder_skips)` # Pass input and skips explicitly
             # `seg_head_input = decoder_out_list[-1]`

             # Modified UnetDecoder2d `__init__`:
             # `encoder_channels_deep_to_shallow`: [C_s3, C_s2, C_s1, C_s0]
             # `decoder_channels_out`: [DC0, DC1, DC2, DC3] (output channels of blocks, deep to shallow)
             # num_blocks = len(decoder_channels_out)
             # Input channels for blocks: [encoder_channels_deep_to_shallow[0], DC0, DC1, DC2]
             # Skip channels for blocks: [encoder_channels_deep_to_shallow[1], encoder_channels_deep_to_shallow[2], encoder_channels_deep_to_shallow[3], 0] # for 4 stages

             if num_decoder_blocks != num_encoder_stages:
                 print(f"Warning: Number of decoder block outputs ({num_decoder_blocks}) does not match number of encoder stages ({num_encoder_stages}). Adjusting number of decoder blocks to match encoder stages.")
                 # Assume decoder_channels specifies output channels for each block corresponding to each encoder stage feature *as a skip*.
                 # The number of decoder blocks should equal the number of encoder features used *as skips* + 1 (for the deepest feature).
                 # If encoder has N stages (0 to N-1), features [s_N-1, ..., s_0] are returned.
                 # s_N-1 is deepest (input to block 0). s_0 is shallowest (skip for block N-1).
                 # Number of decoder blocks = N.
                 # Let's ensure decoder_channels has length N.
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


    # Corrected UnetDecoder2d forward signature and logic:
    # Takes the deepest feature map as main input, and the rest as a list of skips
    def forward(self, deep_feature: torch.Tensor, skip_features: list[torch.Tensor]):
        # deep_feature: output of the deepest encoder stage (s3 in our example)
        # skip_features: list of shallower encoder features [s2, s1, s0]

        x = deep_feature
        decoder_outputs = [x] # Store intermediate decoder block outputs (optional, but maybe useful)

        # Loop through decoder blocks
        for i, block in enumerate(self.blocks):
            # The skip connection for block i comes from skip_features[i]
            skip = skip_features[i] if i < len(skip_features) else None # Handle case where last block has no skip
            x = block(x, skip=skip)
            decoder_outputs.append(x)

        # Return all block outputs or just the final one depending on use case
        # The SegmentationHead takes the *last* decoder block output.
        # Let's return the list of outputs as before, but the last one is the final result.
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

# The original _convnext_block_forward modification might not be needed
# if we are using features_only=True and modifying the stem manually.
# Let's remove it and the replace_forwards call.
# def _convnext_block_forward(self, x):
#     # ... (original timm code) ...
#     pass


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


        # Transformer Encoder Layers
        # Use nn.TransformerEncoder which is a stack of TransformerEncoderLayer
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


        # Add LayerNorm before the transformer blocks if norm_first=False in TransformerEncoderLayer
        # Since norm_first=True, LayerNorm is handled within the layer/encoder.


    def forward(self, x):
        # Input shape: (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.embed_dim, f"Input channel mismatch: {C} != {self.embed_dim}"
        # The input H and W should match the expected sequence lengths,
        # but they might differ slightly due to padding/striding in the stem/backbone.
        # Let's pad the input spatially if needed to match expected seq_len_h/w for PE and transformer.
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


        # Apply positional embeddings (broadcasts over B and the missing spatial dimension)
        # PE shape (1, 1, H, 1, E) -> (B, C, H, W) needs (B, C, H, W) PE
        # PE_h (1, 1, H, 1, E) should be (1, E, H, 1) for 4D tensor adding
        # PE_w (1, 1, 1, W, E) should be (1, E, 1, W) for 4D tensor adding
        # Let's reshape PE params to (1, E, H, 1) and (1, E, 1, W)
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
        backbone: str,
        pretrained: bool = True,
    ):
        super().__init__()

        # Encoder - use features_only=True to get multi-scale features
        # We will manually modify the stem after loading the model.
        self.backbone= timm.create_model(
            backbone,
            in_chans= 5,
            pretrained= pretrained,
            features_only= False, #True, # Get list of features from stages
            drop_path_rate=0.0,
        )

        # Get encoder channel counts from feature_info (deep to shallow)
        # feature_info lists info for each output feature from `features_only=True`
        # The order is typically from the deepest stage to the shallowest.
        encoder_feature_info = self.backbone.feature_info
        encoder_channels = [info['num_chs'] for info in encoder_feature_info] # [C_s3, C_s2, C_s1, C_s0] for a 4-stage model

        # --- Modify the stem AFTER loading ---
        # This requires inspecting the backbone's structure (usually has a 'stem' attribute)
        # The timm ConvNeXt stem is a Sequential block: Conv2d -> LayerNorm2d
        # Let's override the forward method of the backbone to use our custom stem
        # Or, replace the stem layers directly. Replacing layers is cleaner.
        self._update_stem(backbone)

        # --- Determine the spatial size of the last feature map (input to Transformer) ---
        # This depends on the backbone's downsampling after the modified stem.
        # A dummy pass is the most reliable way to get the shapes *after* the custom stem.
        # Or, calculate manually based on kernel/stride/padding for each stage.
        # Let's calculate based on our assumed stem output (71, 72) and standard ConvNeXt downsampling (2x per stage).
        # Stem output: (B, 128, 71, 72) - assuming _update_stem results in this
        # Stages 0, 1, 2, 3 have downsampling before them (except stage 0).
        # Stage 0: no spatial downsample (uses stride 1) -> (71, 72)
        # Stage 1: 2x spatial downsample -> (71//2, 72//2) = (35, 36)
        # Stage 2: 2x spatial downsample -> (35//2, 36//2) = (17, 18)
        # Stage 3: 2x spatial downsample -> (17//2, 18//2) = (8, 9)
        # So the last feature map (from stage 3) is approximately (8, 9).
        # The channel count is encoder_channels[0] (e.g. 768 for convnext_small).

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


    # Helper to calculate stem output shape - only needed if we applied transformer here
    # Or if the subsequent ConvNeXt stages rely on a specific spatial padding/size relationship
    # relative to the stem output size. For now, we assume standard ConvNeXt downsampling applies
    # after the modified stem, and the Transformer input size is calculated accordingly.
    # If spatial sizes are critical, this function might be needed to adjust subsequent layers or padding.
    # def _calculate_stem_output_shape(self, input_shape):
    #     # This is complex due to padding and stride in the custom stem
    #     # A dummy pass through the stem is the most reliable way
    #     with torch.no_grad():
    #         dummy_input = torch.randn(1, *input_shape)
    #         stem_out = self.backbone.stem(dummy_input)
    #         return stem_out.shape[1:] # Return (C, H, W)
    #     pass # Not implemented for now as Transformer is applied later


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

        # Input shape: (B, 5, 1000, 70)
        # Desired shape after stem (approx): (B, C_stem, H_stem, W_stem) where H_stem ~70, W_stem ~70
        # The original code uses two conv layers in the stem modification. Let's replicate that.

        # First conv: adapts channels (5 -> C_stem) and does initial downsampling
        # Original: kernel=(16, 4), stride=(4, 1), padding=(0, 2)
        # Input (1000, 70) -> (247, 71) spatial after this conv
        # The padding and kernel size are unusual for a standard image stem.
        # Let's replicate the structure from the user's original Net class.

        # Note: The original _update_stem replaced the stem_0 (first layer of timm's Sequential stem)
        # and added a second conv. This seems specific. Let's follow the *structure* from the original code.
        # The original code defines stem_0 as nn.Sequential(ReflectionPad2d, original_stem_conv, new_conv)
        # And the original LayerNorm is likely lost unless added back explicitly.
        # Let's redefine the stem Sequential block entirely.

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

        # Where does the LayerNorm go? The original timm stem had Conv->Norm.
        # The original modified stem in user code only showed convs and padding.
        # For ConvNeXt, LayerNorm is crucial after the stem.
        # Let's add a LayerNorm AFTER the two custom conv layers in the new stem.
        # It should normalize over spatial dims (H, W) and channels.
        # LayerNorm(normalized_shape) - should be (C, H, W) or just (C,) depending on usage.
        # ConvNeXt LayerNorm is usually over channels only.
        # LayerNorm(num_features) is equivalent to GroupNorm with group=1 and num_groups=num_features.
        # timm uses LayerNorm(num_channels) with affine=True.
        # The output shape before norm is (B, out_chans, H_final_stem, W_final_stem).
        # Let's add a LayerNorm over the channel dimension.
        # However, `replace_norms` will turn this into InstanceNorm2d.
        # So, just add a standard LayerNorm here, and let `replace_norms` handle it.
        # It should be LayerNorm(out_chans)

        # Let's add the LayerNorm after the second conv
        stem_norm = nn.LayerNorm(out_chans, eps=1e-6) # ConvNeXt uses eps=1e-6
        new_stem_layers.append(stem_norm)


        # Replace the backbone's original stem with the new sequential block
        # timm ConvNeXt usually has backbone.stem as nn.Sequential
        if hasattr(backbone, 'stem') and isinstance(backbone.stem, nn.Sequential):
             backbone.stem = nn.Sequential(*new_stem_layers)
             print(f"Replaced backbone stem with custom sequential block.")
             # Also need to modify the first downsample layer, as the stem output
             # spatial size is different from the original ConvNeXt expected input to stages[0].
             # Standard ConvNeXt has a downsample layer *after* the stem and *before* stage 0.
             # This downsample layer usually has stride 1 and changes channels.
             # In features_only=True, stage 0 is the first element in `stages`.
             # The input to stages[0] is the output of the stem.
             # The first *spatial* downsampling (2x) happens *before* stage 1, via downsample_layers[0].
             # The custom stem already aggressively downsamples H (1000 -> 71).
             # Standard stages/downsampling might need adjustment.
             # Let's assume the standard downsample_layers and stages can handle the (71, 72) spatial input from the stem.
             # If not, more complex surgery on the backbone is needed, which violates 'lightweight'.
             # Stick to modifying only the stem and applying Transformer after the last stage.
        else:
             print(f"Warning: Backbone stem structure is not as expected for modification.")


    def replace_activations(self, module, log=False):
        """ Recursively replaces specific activation functions with GELU. """
        if log:
            print(f"Replacing activations with GELU...")

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
            print(f"Replacing norms with InstanceNorm2d (skipping LayerNorm)...")

        for name, c in mod.named_children():
            # Skip LayerNorm as it's used in Transformers and ConvNeXt often uses it after stem
            if isinstance(c, nn.LayerNorm):
                if log: print(f"  Skipping LayerNorm at {name}")
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
                if log: print(f"  Replacing {type(c).__name__} at {name} with InstanceNorm2d")
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


        # The backbone was loaded with features_only=True, but we replaced the stem.
        # The backbone forward *might* still call its internal original stem logic or expect a certain input format.
        # It's safer to call the stages manually after our modified stem.

        # Manual backbone forward
        # Need access to backbone.stem, backbone.stages (ModuleList), backbone.downsample_layers (ModuleList)
        # Check if the modified stem and these attributes exist
        if not hasattr(self.backbone, 'stem') or not hasattr(self.backbone, 'stages') or not hasattr(self.backbone, 'downsample_layers'):
            raise AttributeError("Backbone structure not as expected for manual forward pass.")

        with torch.autocast(device_type=self.cfg.device.type, dtype=amp_dtype, enabled=amp_enabled):
             x_in = x # Keep original input for inference-time flip

             # Pass through the custom stem
             # The stem expects (B, C, H, W) where C=5, H=1000, W=70
             stem_out = self.backbone.stem(x) # Output (B, C_stem, H_stem, W_stem) e.g. (B, 128, 71, 72)

             # Pass through subsequent stages and downsamples
             # Collect intermediate features for the decoder
             features_for_decoder = []
             # Stage 0 is the first stage after the stem. Add its output.
             x = self.backbone.stages[0](stem_out)
             features_for_decoder.append(x) # s0

             # Pass through subsequent stages and downsamples
             # Stages 1, 2, 3 are connected via downsample layers
             # The i-th downsample_layer is applied before the i+1-th stage
             for i in range(len(self.backbone.downsample_layers)): # Loop through downsample layers
                 # downsample_layers[i] is applied to the output of stages[i]
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

             # Post-processing (cropping, scaling)
             # The output is (B, 1, 72, 72) approximately due to padding, need to crop to 70x70
             # Assumes the output size is 72x72 based on stem calculation and no further downsampling affecting this ratio.
             # Need to calculate the actual output size reliably or make cropping dynamic.
             # Let's assume it's 72x72 and crop 1 pixel from each side.
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


# --- Data Loading ---
# Make sure file_pairs list is not empty before creating datasets
if not cfg.file_pairs:
    print("No data file pairs found. Exiting.")
    # You would typically handle this case appropriately, e.g., exit, raise error, or load dummy data
    # For this notebook structure, we'll just print and the subsequent code might fail
    train_dl = [] # Empty lists to prevent errors later
    valid_dl = []
    print("Created empty dataloaders.")
else:
    # Use a small subset for validation/test if subsample is used for train
    # Or split files into train/val/test sets. For simplicity, use all files for both train/val here.
    # In a real scenario, you'd split `cfg.file_pairs` into train_files, val_files, test_files
    # and pass the appropriate list to each dataset instance.
    # For demonstration, using all files for train and val as in the original code structure,
    # but the subsample logic will limit the actual number of samples.

    # Small split for demonstration purposes
    total_files = len(cfg.file_pairs)
    print(f"number of files in the data base: {total_files}")
    if total_files > 1:
        split_idx = max(1, int(total_files * 0.8)) # 80% train, 20% val
        train_file_pairs = cfg.file_pairs[:split_idx]
        valid_file_pairs = cfg.file_pairs[split_idx:]
        # Optional: Further split valid_file_pairs for a separate test set
        # test_file_pairs = ...
    else:
        # If only one file, use it for both (not ideal for training)
        train_file_pairs = cfg.file_pairs
        valid_file_pairs = cfg.file_pairs


    print(f"Using {len(train_file_pairs)} files for training and {len(valid_file_pairs)} for validation.")

    train_ds = CustomDataset(cfg=cfg, file_pairs=train_file_pairs, mode="train")
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size= cfg.batch_size,
        num_workers= 0, # Set to > 0 for faster loading if memory allows
        shuffle=True,
        pin_memory=True, # speeds up data transfer to GPU
    )

    valid_ds = CustomDataset(cfg=cfg, file_pairs=valid_file_pairs, mode="valid")
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size= cfg.batch_size_val,
        num_workers= 0, # Set to > 0 for faster loading
        shuffle=False,
        pin_memory=True,
    )

# Check dataset output shapes
if train_dl:
    x, y = next(iter(train_dl))
    print("\nDataLoader sample shapes:")
    print("Input (x):", x.shape) # Expected: (B, C, T, W) -> (B, 5, 1000, 70)
    print("Output (y):", y.shape) # Expected: (B, 1, H', W') -> (B, 1, 70, 70)


# ========== Model / Optim ==========
print(f"\nInitializing model on device: {cfg.device}")
model = Net(backbone=cfg.backbone).to(cfg.device)

# Check initial model output shape with a dummy input on the correct device
if train_dl:
    try:
        dummy_input = torch.randn(cfg.batch_size, 5, 1000, 70).to(cfg.device)
        with torch.no_grad():
             dummy_output = model(dummy_input)
        print(f"Dummy model output shape: {dummy_output.shape}")
        expected_output_shape = (cfg.batch_size, 1, 70, 70)
        if dummy_output.shape != expected_output_shape:
             print(f"Warning: Model output shape {dummy_output.shape} does not match expected {expected_output_shape}.")
    except Exception as e:
        print(f"Error during dummy model forward pass: {e}")


if cfg.ema:
    if cfg.local_rank == 0:
        print("Initializing EMA model..")
    # Initialize EMA model on the same device as the main model
    ema_model = ModelEMA(
        model,
        decay=cfg.ema_decay,
        device=cfg.device,
    )
else:
    ema_model = None

# criterion = nn.L1Loss() # Original criterion
# Common losses for regression maps: MSELoss, L1Loss, HuberLoss. L1 is less sensitive to outliers.
# For map-like output, often MSE or smooth L1 are used.
criterion = nn.MSELoss() # Let's use MSELoss as it's common for regression

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# Removed GradScaler - Re-add if using float16 AMP during training for stability
# If using bfloat16, a scaler is often not strictly necessary but can still help in some cases.
# The current code does NOT use a scaler with AMP. This might be unstable with float16.
# For bfloat16, it's usually okay.
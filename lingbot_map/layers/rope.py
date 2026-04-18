# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

# This module provides a clean implementation of 2D Rotary Position Embeddings,
# which extends the original RoPE concept to handle 2D spatial positions.

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from typing import List, Optional, Tuple, Union


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid.

    This class efficiently manages the generation of spatial coordinates for patches
    in a 2D grid, caching results to avoid redundant computations.

    Attributes:
        position_cache: Dictionary storing precomputed position tensors for different
            grid dimensions.
    """

    def __init__(self):
        """Initializes the position generator with an empty cache."""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Generates spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of the grid in patches.
            width: Width of the grid in patches.
            device: Target device for the position tensor.

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
            for each position in the grid, repeated for each batch item.
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation.

    This module applies rotary position embeddings to input tokens based on their
    2D spatial positions. It handles the position-dependent rotation of features
    separately for vertical and horizontal dimensions.

    Args:
        frequency: Base frequency for the position embeddings. Default: 100.0
        scaling_factor: Scaling factor for frequency computation. Default: 1.0

    Attributes:
        base_frequency: Base frequency for computing position embeddings.
        scaling_factor: Factor to scale the computed frequencies.
        frequency_cache: Cache for storing precomputed frequency components.
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """Initializes the 2D RoPE module."""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes frequency components for rotary embeddings.

        Args:
            dim: Feature dimension (must be even).
            seq_len: Maximum sequence length.
            device: Target device for computations.
            dtype: Data type for the computed tensors.

        Returns:
            Tuple of (cosine, sine) tensors for frequency components.
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # Compute and cache frequency components
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """Performs feature rotation by splitting and recombining feature dimensions.

        Args:
            x: Input tensor to rotate.

        Returns:
            Rotated feature tensor.
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """Applies 1D rotary position embeddings along one dimension.

        Args:
            tokens: Input token features.
            positions: Position indices.
            cos_comp: Cosine components for rotation.
            sin_comp: Sine components for rotation.

        Returns:
            Tokens with applied rotary position embeddings.
        """
        # Embed positions with frequency components
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # Apply rotation
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Applies 2D rotary position embeddings to input tokens.

        Args:
            tokens: Input tensor of shape (batch_size, n_heads, n_tokens, dim).
                   The feature dimension (dim) must be divisible by 4.
            positions: Position tensor of shape (batch_size, n_tokens, 2) containing
                      the y and x coordinates for each token.

        Returns:
            Tensor of same shape as input with applied 2D rotary position embeddings.

        Raises:
            AssertionError: If input dimensions are invalid or positions are malformed.
        """
        # Validate inputs
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        # Compute feature dimension for each spatial direction
        feature_dim = tokens.size(-1) // 2

        # Get frequency components
       max_position = (positions.max() + 1).to(dtype=torch.int64)

cos_comp, sin_comp = self._compute_frequency_components(
    feature_dim,
    max_position.item(),
    tokens.device,
    tokens.dtype,
)

        # Split features for vertical and horizontal processing
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # Apply RoPE separately for each dimension
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # Combine processed features
        return torch.cat((vertical_features, horizontal_features), dim=-1)
    


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    计算1D旋转位置编码（RoPE）的频率张量。
    
    RoPE的核心思想：使用旋转矩阵来编码位置信息，使得相对位置关系保持不变。
    公式：对于位置m和维度i，频率为 θ_i = θ^(-2i/d)，其中θ是基础频率（默认10000）
    
    Args:
        dim: 特征维度，必须是偶数（因为要成对处理）
        pos: 位置索引，可以是整数（自动生成0到pos-1的序列）或位置数组 [S]
        theta: 基础频率，控制位置编码的周期性（默认10000）
        use_real: 是否返回实数形式（cos和sin分开）还是复数形式
        linear_factor: 线性缩放因子，用于上下文扩展
        ntk_factor: NTK-Aware缩放因子，用于处理更长的序列
        repeat_interleave_real: 当use_real=True时，是否交错重复（用于某些模型架构）
        freqs_dtype: 频率张量的数据类型
        
    Returns:
        复数形式：[S, D/2] 的复数张量，表示 e^(i*m*θ_j)
        实数形式：两个 [S, D] 的张量（cos和sin）
    """
    # 确保维度是偶数（RoPE需要成对处理维度）
    assert dim % 2 == 0

    # 将位置转换为torch张量
    if isinstance(pos, int):
        pos = torch.arange(pos)  # 生成 [0, 1, 2, ..., pos-1]
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # [S]

    # 应用NTK缩放（Neural Tangent Kernel，用于处理训练时未见过的长序列）
    theta = theta * ntk_factor
    
    # 步骤1：计算频率 θ_i = 1 / (θ^(2i/d))
    # 其中 i ∈ {0, 2, 4, ..., dim-2}（只取偶数索引，因为成对处理）
    # 公式：freq_i = 1 / (theta^(2i/d) * linear_factor)
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]，每个频率对应一个维度对
    
    # 步骤2：计算位置-频率矩阵
    # 使用外积：pos[m] * freqs[i] = m * θ_i
    # 结果：每个位置m和每个频率i的组合
    freqs = torch.outer(pos, freqs)  # [S, D/2]
    
    # 步骤3：根据返回格式转换
    if use_real and repeat_interleave_real:
        # 方式1：交错重复（用于flux, hunyuan-dit, cogvideox等模型）
        # 将每个频率的cos和sin交错排列：[cos_0, cos_0, cos_1, cos_1, ...]
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # 方式2：拼接重复（用于stable audio, allegro等模型）
        # 将所有cos拼接，然后是所有sin：[cos_0, cos_1, ..., cos_n, cos_0, cos_1, ..., cos_n]
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # 方式3：复数形式（用于lumina等模型）
        # 使用欧拉公式：e^(iθ) = cos(θ) + i*sin(θ)
        # torch.polar(r, θ) 返回 r * e^(iθ)，这里r=1，所以就是 e^(i*freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64: [S, D/2]
        return freqs_cis


class WanRotaryPosEmbed(nn.Module):
    """
    3D旋转位置编码（3D RoPE）模块
    
    核心思想：将RoPE扩展到3D空间（时间、高度、宽度），为视频或3D数据提供位置编码。
    每个维度（t, h, w）独立使用RoPE，然后拼接起来。
    
    公式：
    对于3D位置 (f, h, w)（帧、高度、宽度）：
    - 帧维度使用 dim_f 个特征维度
    - 高度维度使用 dim_h 个特征维度  
    - 宽度维度使用 dim_w 个特征维度
    其中 dim_f + dim_h + dim_w = attention_head_dim
    """
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
        fhw_dim: Optional[Tuple[int, int, int]] = [20, 22, 22],
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim  # 注意力头的总维度
        self.patch_size = patch_size  # patch大小 (patch_f, patch_h, patch_w)
        self.max_seq_len = max_seq_len  # 最大序列长度（用于预计算频率）

        # 步骤1：分配维度给三个空间维度
        if fhw_dim is not None:
            # 如果指定了维度分配，使用指定的
            assert attention_head_dim == sum(
                fhw_dim
            ), f"attention_head_dim {attention_head_dim} must match sum(fhw_dim) {sum(fhw_dim)}"
            t_dim, h_dim, w_dim = fhw_dim
        else:
            # 否则自动分配：h和w各占1/3，t占剩余
            # 例如：如果attention_head_dim=64，则 h_dim=w_dim=21，t_dim=22
            h_dim = w_dim = 2 * (attention_head_dim // 6)
            t_dim = attention_head_dim - h_dim - w_dim
        
        # 保存维度分配以便在forward中使用
        self.fhw_dim = (t_dim, h_dim, w_dim)

        # 步骤2：为每个维度预计算频率
        # 分别计算时间、高度、宽度三个维度的RoPE频率
        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            # 每个维度独立调用1D RoPE
            # 返回复数形式的频率: [max_seq_len, dim//2]
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        # 将三个维度的频率在最后一维拼接: [max_seq_len, (t_dim + h_dim + w_dim)//2]
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, ppf, pph, ppw, patch_start_idx, device: torch.device, f_start: int = 0, f_end: Optional[int] = None) -> torch.Tensor:
        """
        前向传播：为3D输入（视频帧+patch）生成旋转位置编码
        
        参数：
        - ppf (int): 帧数（patches per frame），当f_end为None时使用
        - pph (int): 每帧的patch高度数量
        - ppw (int): 每帧的patch宽度数量  
        - patch_start_idx (int): 每帧的特殊token数量（在patches之前）
        - device: 计算设备（CPU/GPU）
        - f_start (int): 起始帧索引（用于causal模式），默认为0
        - f_end (Optional[int]): 结束帧索引（用于causal模式），如果为None则使用ppf作为帧数
        
        返回：
        - freqs: [1, 1, ppf * (patch_start_idx + pph * ppw), head_dim//2] 复数频率tensor
        
        Token排列顺序：
        [frame0_special_token_0, ..., frame0_special_token_N,
         frame0_patch_0, ..., frame0_patch_M,
         frame1_special_token_0, ..., frame1_special_token_N,
         frame1_patch_0, ..., frame1_patch_M,
         ...]
        
        模式：
        - 非causal模式：f_end=None，使用ppf作为帧数，从位置0开始
        - Causal模式：f_end不为None，使用[f_start, f_end)范围的帧，ppf会被重新计算
        """

        # 步骤1：将预计算的频率移到目标设备，并分割成三个维度
        self.freqs = self.freqs.to(device)
        # 获取实际的维度分配
        if hasattr(self, 'fhw_dim') and self.fhw_dim is not None:
            t_dim, h_dim, w_dim = self.fhw_dim
        else:
            # 自动分配的情况
            h_dim = w_dim = 2 * (self.attention_head_dim // 6)
            t_dim = self.attention_head_dim - h_dim - w_dim
        
        # 使用正确的split sizes（每个维度的一半）
        freqs = self.freqs.split_with_sizes(
            [
                t_dim // 2,  # 时间维度
                h_dim // 2,  # 高度维度
                w_dim // 2,  # 宽度维度
            ],
            dim=1,
        )
        
        # 处理causal模式：如果指定了f_end，重新计算ppf和帧范围
        if f_end is not None:
            ppf = f_end - f_start
            frame_slice = slice(f_start, f_end)
        else:
            # 非causal模式：使用从0开始的ppf个帧
            frame_slice = slice(0, ppf)
        
        # 步骤2：处理特殊token（如果存在）
        ## For other tokens
        if patch_start_idx > 0:
            # 2.1 为特殊token生成位置编码
            # 特殊token位于对角线位置 (f, i, i)，每个特殊token有唯一位置
            # camera: (f, 0, 0), register_0: (f, 1, 1), ..., scale: (f, 5, 5)
            # Shape: (ppf, patch_start_idx, dim)
            freqs_special_f = freqs[0][frame_slice].reshape(ppf, 1, -1).expand(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim_f) 帧维度变化
            freqs_special_h = freqs[1][:patch_start_idx].reshape(1, patch_start_idx, -1).expand(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim_h) 高度=0,1,2,...
            freqs_special_w = freqs[2][:patch_start_idx].reshape(1, patch_start_idx, -1).expand(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim_w) 宽度=0,1,2,...
            freqs_special = torch.cat([freqs_special_f, freqs_special_h, freqs_special_w], dim=-1)  # (ppf, patch_start_idx, dim) 拼接三维
            freqs_special = freqs_special.reshape(ppf, patch_start_idx, -1)  # (ppf, patch_start_idx, dim)

            # 2.2 为图像patch生成位置编码
            # Patch位于 (f, patch_start_idx+h, patch_start_idx+w)，h,w 整体偏移 patch_start_idx
            # 这样 patches 与 special tokens 位置不冲突，且 h,w 对称处理
            # Shape: (ppf, pph, ppw, dim)
            freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_f) 帧维度
            freqs_h = freqs[1][patch_start_idx : patch_start_idx + pph].reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_h) 高度从patch_start_idx开始
            freqs_w = freqs[2][patch_start_idx : patch_start_idx + ppw].reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_w) 宽度从patch_start_idx开始
            freqs_patches = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)  # (ppf, pph, ppw, dim) 拼接三维
            freqs_patches = freqs_patches.reshape(ppf, pph * ppw, -1)  # (ppf, pph * ppw, dim) 展平空间维度
            
            # 步骤3：按照正确的顺序组合特殊token和patches
            # 每帧内部顺序：[特殊tokens, patches]
            # Concatenate special tokens and patches for each frame along the second dimension
            # Shape: (ppf, patch_start_idx + pph * ppw, dim)
            freqs = torch.cat([freqs_special, freqs_patches], dim=1)  # (ppf, patch_start_idx + pph * ppw, dim)
            
            # 步骤4：展平为最终形状并添加batch和head维度
            # Flatten to get final shape: (ppf * (patch_start_idx + pph * ppw), dim)
            freqs = freqs.reshape(ppf * (patch_start_idx + pph * ppw), -1)
            freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, ppf * (patch_start_idx + pph * ppw), dim) 添加batch和head维度
            return freqs
        
        # 如果没有特殊token（patch_start_idx == 0），只处理图像patches
        # 所有patches位于 (f, 0:pph, 0:ppw)
        freqs_f = freqs[0][frame_slice].reshape(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_f) 帧维度
        freqs_h = freqs[1][:pph].reshape(1, pph, 1, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_h) 高度从0开始
        freqs_w = freqs[2][:ppw].reshape(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)  # (ppf, pph, ppw, dim_w) 宽度从0开始
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)  # (1, 1, ppf * pph * ppw, dim)
        return freqs
    
def apply_rotary_emb(x, freqs):
    """
    应用旋转位置编码到输入特征
    
    核心思想：使用复数乘法实现特征旋转，保持相对位置信息
    
    数学原理：
    对于2D向量 [x1, x2]，旋转θ角度可以表示为复数乘法：
    (x1 + ix2) * e^(iθ) = (x1 + ix2) * (cos(θ) + i*sin(θ))
                        = (x1*cos(θ) - x2*sin(θ)) + i*(x1*sin(θ) + x2*cos(θ))
    
    这等价于旋转矩阵：
    [cos(θ)  -sin(θ)] [x1]
    [sin(θ)   cos(θ)] [x2]
    
    参数：
    - x: 输入特征 [batch, heads, seq_len, head_dim]
    - freqs: 旋转频率（复数） [1, 1, seq_len, head_dim//2]
    
    返回：
    - x_out: 旋转后的特征 [batch, heads, seq_len, head_dim]
    
    实现步骤：
    1. 将x的每两个连续特征看作一个复数 (real, imag)
    2. 与预计算的复数频率 e^(iθ) 相乘
    3. 转回实数表示
    """
    # 步骤1：reshape成 [..., head_dim//2, 2] 形式，最后一维表示(real, imag)
    # 例如：[b, h, seq, 64] -> [b, h, seq, 32, 2]
    x_reshaped = x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    
    # 步骤2：转换为复数表示 [b, h, seq, 32]
    # 每个元素是 real + imag*i
    x_complex = torch.view_as_complex(x_reshaped)
    
    # 步骤3：复数乘法实现旋转
    # x_complex * freqs 相当于将每对特征旋转θ角度
    # freqs已经是 e^(iθ) = cos(θ) + i*sin(θ) 的形式
    x_rotated = x_complex * freqs
    
    # 步骤4：转回实数表示 [b, h, seq, 32, 2]
    x_real = torch.view_as_real(x_rotated)
    
    # 步骤5：展平最后两维 [b, h, seq, 64]
    x_out = x_real.flatten(3)
    
    # 步骤6：转回原始数据类型
    return x_out.to(x.dtype)

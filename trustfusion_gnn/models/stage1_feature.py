"""
Stage 1: 特征提取与初始可信度估计
对应你架构图中的：
- Temporal Encoder (1D-CNN/GRU) → h_temp
- Statistical Feature Extraction → s_feat  
- Initial Credibility Estimation → τ_init
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class TemporalEncoder(nn.Module):
    """
    时序编码器
    输入: X ∈ ℝ^{N×T×F}
    输出: h_temp ∈ ℝ^{N×T×H}
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 多尺度时序卷积
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                         padding=padding, dilation=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        # GRU for long-term dependencies
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, T, F)
        Returns:
            h_temp: (batch, N, T, H)
        """
        B, N, T, F = x.shape
        
        # 展平 batch 和 nodes
        x = x.view(B * N, T, F)
        
        # 输入投影
        x = self.input_proj(x)  # (B*N, T, H)
        
        # 时序卷积（需要转置）
        x_conv = x.transpose(1, 2)  # (B*N, H, T)
        for conv in self.convs:
            x_conv = x_conv + conv(x_conv)  # 残差连接
        x_conv = x_conv.transpose(1, 2)  # (B*N, T, H)
        
        # GRU
        x_gru, _ = self.gru(x_conv)  # (B*N, T, H)
        
        # 残差 + LayerNorm
        h_temp = self.output_norm(x_conv + x_gru)
        
        # 恢复形状
        h_temp = h_temp.view(B, N, T, -1)
        
        return h_temp


class StatisticalFeatureExtractor(nn.Module):
    """
    统计特征提取器
    从原始时序中提取统计特征：均值、方差、偏度、峰度、变化率等
    输出: s_feat ∈ ℝ^{N×S}
    """
    
    def __init__(self, num_features: int = 8):
        super().__init__()
        self.num_features = num_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, T, F) 原始观测
        Returns:
            s_feat: (batch, N, num_features)
        """
        B, N, T, F = x.shape
        
        # 沿时间维度计算统计量
        x_squeezed = x.squeeze(-1)  # (B, N, T) 假设 F=1
        
        features = []
        
        # 1. 均值
        mean = x_squeezed.mean(dim=-1)  # (B, N)
        features.append(mean)
        
        # 2. 标准差
        std = x_squeezed.std(dim=-1)  # (B, N)
        features.append(std)
        
        # 3. 最大值
        max_val = x_squeezed.max(dim=-1)[0]
        features.append(max_val)
        
        # 4. 最小值
        min_val = x_squeezed.min(dim=-1)[0]
        features.append(min_val)
        
        # 5. 变化率（一阶差分的均值绝对值）
        diff = torch.diff(x_squeezed, dim=-1)
        change_rate = diff.abs().mean(dim=-1)
        features.append(change_rate)
        
        # 6. 峰值数量（简化：局部最大值数量）
        # 使用相邻差分的符号变化
        diff_sign = torch.sign(diff)
        sign_change = torch.diff(diff_sign, dim=-1)
        peaks = (sign_change == -2).float().sum(dim=-1) / T
        features.append(peaks)
        
        # 7. 偏度（三阶矩）
        mean_expanded = mean.unsqueeze(-1)
        std_expanded = std.unsqueeze(-1) + 1e-8
        normalized = (x_squeezed - mean_expanded) / std_expanded
        skewness = (normalized ** 3).mean(dim=-1)
        features.append(skewness)
        
        # 8. 峰度（四阶矩）
        kurtosis = (normalized ** 4).mean(dim=-1) - 3
        features.append(kurtosis)
        
        # 堆叠所有特征
        s_feat = torch.stack(features[:self.num_features], dim=-1)  # (B, N, S)
        
        return s_feat


class InitialCredibilityEstimator(nn.Module):
    """
    初始可信度估计器
    基于统计特征估计初始可信度 τ_init
    """
    
    def __init__(
        self,
        temporal_dim: int = 64,
        statistical_dim: int = 8,
        hidden_dim: int = 32
    ):
        super().__init__()
        
        # 融合时序特征和统计特征
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(temporal_dim + statistical_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        h_temp: torch.Tensor,
        s_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_temp: (batch, N, T, H) 时序特征
            s_feat: (batch, N, S) 统计特征
        Returns:
            tau_init: (batch, N) 初始可信度
        """
        B, N, T, H = h_temp.shape
        
        # 时序特征池化
        h_temp_flat = h_temp.view(B * N, T, H).transpose(1, 2)  # (B*N, H, T)
        h_pooled = self.temporal_pool(h_temp_flat).squeeze(-1)  # (B*N, H)
        h_pooled = h_pooled.view(B, N, H)  # (B, N, H)
        
        # 拼接特征
        combined = torch.cat([h_pooled, s_feat], dim=-1)  # (B, N, H+S)
        
        # 估计初始可信度
        tau_init = self.mlp(combined).squeeze(-1)  # (B, N)
        
        return tau_init


class Stage1Module(nn.Module):
    """
    Stage 1 完整模块
    输入: X ∈ ℝ^{N×T×F}
    输出: h_temp, s_feat, τ_init
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        statistical_features: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            dropout=dropout
        )
        
        self.statistical_extractor = StatisticalFeatureExtractor(
            num_features=statistical_features
        )
        
        self.credibility_estimator = InitialCredibilityEstimator(
            temporal_dim=hidden_dim,
            statistical_dim=statistical_features
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, N, T, F)
        Returns:
            h_temp: (batch, N, T, H)
            s_feat: (batch, N, S)
            tau_init: (batch, N)
        """
        # 时序编码
        h_temp = self.temporal_encoder(x)
        
        # 统计特征
        s_feat = self.statistical_extractor(x)
        
        # 初始可信度
        tau_init = self.credibility_estimator(h_temp, s_feat)
        
        return h_temp, s_feat, tau_init
"""
数据结构定义
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from enum import Enum
import torch


@dataclass
class SystemInput:
    """
    系统输入 X = {X, A, M}
    对应你架构图中的：
    1. 多传感器观测序列 X ∈ ℝ^{N×T×F}
    2. 传感器空间关系图 A ∈ ℝ^{N×N}
    3. 传感器元信息 M ∈ ℝ^{N×D}
    """
    # 观测数据 X ∈ ℝ^{N×T×F}
    X: torch.Tensor                    # shape: (batch, N, T, F) or (N, T, F)
    
    # 邻接矩阵 A ∈ ℝ^{N×N}
    A: torch.Tensor                    # shape: (N, N)
    
    # 元信息嵌入 M ∈ ℝ^{N×D}
    M: Optional[torch.Tensor] = None   # shape: (N, D)
    
    # 附加信息
    timestamps: Optional[np.ndarray] = None
    sensor_ids: Optional[List[str]] = None

@dataclass
class SystemOutput:
    """系统输出"""
    Y_hat: torch.Tensor                # (batch, T, F) 融合结果
    tau: torch.Tensor                  # (batch, N) 传感器可信度
    tau_full: torch.Tensor             # (batch, N, T) 时变可信度
    anomaly_flags: torch.Tensor        # (batch, N) 异常标识
    anomaly_scores: torch.Tensor       # (batch, N) 异常分数
    sigma: torch.Tensor                # (batch, T, F) 不确定性
    system_confidence: torch.Tensor    # (batch,) 系统置信度
    node_embeddings: Optional[torch.Tensor] = None
    learned_adjacency: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None

# ==================== Stage 中间输出 ====================

@dataclass  
class Stage1Output:
    """
    Stage 1 输出
    对应你架构图中的：h_temp, s_feat, τ_init
    """
    h_temp: torch.Tensor      # 时序特征 (batch, N, T, H)
    s_feat: torch.Tensor      # 统计特征 (batch, N, S)
    tau_init: torch.Tensor    # 初始可信度 (batch, N)


@dataclass
class Stage2Output:
    """
    Stage 2 输出
    对应你架构图中的：多跳消息传递后的节点嵌入
    """
    node_embeddings: torch.Tensor    # (batch, N, T, H)
    attention_weights: torch.Tensor  # (N, N) 注意力权重
    learned_adj: torch.Tensor        # (N, N) 学习到的邻接矩阵

@dataclass
class FusionResult:
    """应用层融合结果"""
    timestamp: datetime
    fused_values: Dict[str, float]
    uncertainties: Dict[str, float]
    sensor_credibility: Dict[str, float]
    anomaly_flags: Dict[str, bool]
    fusion_weights: Dict[str, float]
    system_confidence: float
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
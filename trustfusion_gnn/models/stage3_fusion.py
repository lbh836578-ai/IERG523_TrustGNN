"""
Stage 3: 可信度精炼与融合输出
对应你架构图中的：
- Credibility Refinement → τ
- Weighted Aggregation → Ŷ  
- Uncertainty Estimation → σ
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class CredibilityRefinement(nn.Module):
    """
    可信度精炼模块
    综合时序信息和邻居一致性精炼可信度
    """
    
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        
        # 时序可信度聚合
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # 精炼网络
        self.refine_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau_temporal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: (batch, N, T, H) 节点嵌入
            tau_temporal: (batch, N, T) 时变可信度
            
        Returns:
            tau_final: (batch, N) 最终可信度（每个传感器一个值）
        """
        B, N, T, H = h.shape
        
        # 对每个节点，用注意力聚合时序信息
        h_flat = h.view(B * N, T, H)
        h_attended, _ = self.temporal_attention(h_flat, h_flat, h_flat)
        h_pooled = h_attended.mean(dim=1)  # (B*N, H)
        h_pooled = h_pooled.view(B, N, H)
        
        # 时变可信度的统计量
        tau_mean = tau_temporal.mean(dim=-1, keepdim=True)  # (B, N, 1)
        
        # 精炼
        combined = torch.cat([h_pooled, tau_mean], dim=-1)
        tau_final = self.refine_net(combined).squeeze(-1)  # (B, N)
        
        return tau_final


class WeightedAggregation(nn.Module):
    """
    可信度加权聚合
    Ŷ = Σ (τ_i / Σ τ_j) · f(h_i)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_sensors: int,
        output_dim: int = 4,
        sensor_to_output: Dict[int, int] = None
    ):
        """
        Args:
            sensor_to_output: 映射每个传感器到输出通道的字典
                             例如 {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3}
                             表示传感器0,2贡献到输出0（温度）等
        """
        super().__init__()
        
        self.num_sensors = num_sensors
        self.output_dim = output_dim
        
        # 默认映射：按顺序分配
        if sensor_to_output is None:
            # 假设: 0,1→温度, 2,3→湿度, 4,5→土壤, 6→光照
            sensor_to_output = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3}
        self.register_buffer(
            'sensor_to_output', 
            torch.tensor([sensor_to_output.get(i, 0) for i in range(num_sensors)])
        )
        
        # 每个传感器的输出投影
        self.sensor_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, T, H) 节点嵌入
            tau: (batch, N) 可信度
            
        Returns:
            y_hat: (batch, T, output_dim) 融合结果
            weights: (batch, N, output_dim) 实际使用的权重
        """
        B, N, T, H = h.shape
        
        # 传感器输出贡献
        sensor_out = self.sensor_proj(h)  # (B, N, T, output_dim)
        
        # 计算每个输出通道的加权融合（避免原地赋值）
        y_hat_channels = []
        weight_channels = []
        
        for out_idx in range(self.output_dim):
            # 找到贡献到这个输出的传感器
            mask = (self.sensor_to_output == out_idx)  # (N,)
            
            if mask.sum() == 0:
                y_hat_channels.append(torch.zeros(B, T, device=h.device, dtype=h.dtype))
                weight_channels.append(torch.zeros(B, N, device=h.device, dtype=h.dtype))
                continue
                
            # 提取对应传感器的可信度和输出
            tau_masked = tau[:, mask]  # (B, num_contributing)
            out_masked = sensor_out[:, mask, :, out_idx]  # (B, num_contributing, T)
            
            # 归一化权重
            w = tau_masked / (tau_masked.sum(dim=1, keepdim=True) + 1e-8)  # (B, num_contributing)
            
            # 加权聚合
            weighted_out = (out_masked * w.unsqueeze(-1)).sum(dim=1)  # (B, T)
            y_hat_channels.append(weighted_out)

            # 记录权重（映射回 N 维）
            contributing_idx = torch.where(mask)[0]  # (K,)
            selector = F.one_hot(contributing_idx, num_classes=N).to(dtype=h.dtype, device=h.device)  # (K, N)
            weights_full = torch.matmul(w, selector)  # (B, N)
            weight_channels.append(weights_full)

        y_hat = torch.stack(y_hat_channels, dim=-1)  # (B, T, output_dim)
        weights = torch.stack(weight_channels, dim=-1)  # (B, N, output_dim)
        
        return y_hat, weights


class UncertaintyEstimation(nn.Module):
    """
    不确定性估计
    σ = f(h, τ, var(sensor_outputs))
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + output_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # 确保正值
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        y_hat: torch.Tensor,
        sensor_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: (batch, N, T, H)
            tau: (batch, N)
            y_hat: (batch, T, output_dim) 融合结果
            sensor_outputs: (batch, N, T, output_dim) 各传感器的输出
            
        Returns:
            sigma: (batch, T, output_dim) 不确定性
        """
        B, N, T, H = h.shape
        output_dim = y_hat.shape[-1]
        
        # 全局特征
        h_global = h.mean(dim=(1, 2))  # (B, H)
        
        # 平均可信度
        tau_mean = tau.mean(dim=1, keepdim=True)  # (B, 1)
        
        # 传感器输出的方差（反映不一致性）
        sensor_var = sensor_outputs.var(dim=1).mean(dim=1)  # (B, output_dim)
        
        # 拼接特征
        combined = torch.cat([h_global, tau_mean, sensor_var], dim=-1)  # (B, H + 1 + output_dim)
        
        # 估计基础不确定性
        base_sigma = self.mlp(combined)  # (B, output_dim)
        
        # 扩展到所有时间步
        sigma = base_sigma.unsqueeze(1).expand(-1, T, -1)  # (B, T, output_dim)
        
        return sigma


class AnomalyDetector(nn.Module):
    """
    异常检测模块
    综合多种信号判断传感器是否异常
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_criteria: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 多准则异常检测
        self.criteria_weights = nn.Parameter(torch.ones(num_criteria) / num_criteria)
        
        # 从嵌入预测异常
        self.anomaly_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        raw_input: torch.Tensor,
        neighbor_consistency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, T, H) 节点嵌入
            tau: (batch, N) 可信度
            raw_input: (batch, N, T, F) 原始输入
            neighbor_consistency: (batch, N) 与邻居的一致性分数
            
        Returns:
            anomaly_scores: (batch, N) 异常分数 [0, 1]
            anomaly_flags: (batch, N) 异常标识 {0, 1}
        """
        B, N, T, H = h.shape
        
        # 准则1: 基于嵌入的异常预测
        h_pooled = h.mean(dim=2)  # (B, N, H)
        score_embedding = self.anomaly_net(h_pooled).squeeze(-1)  # (B, N)
        
        # 准则2: 低可信度
        score_credibility = 1 - tau  # 可信度低 = 更可能异常
        
        # 准则3: 与邻居不一致
        score_consistency = 1 - neighbor_consistency
        
        # 准则4: 时序变化异常（变化率过大或过小）
        diff = torch.diff(raw_input.squeeze(-1), dim=2)
        change_rate = diff.abs().mean(dim=2)  # (B, N)
        # 归一化到 [0, 1]
        score_temporal = torch.sigmoid(change_rate - change_rate.mean(dim=1, keepdim=True))
        
        # 加权组合
        weights = F.softmax(self.criteria_weights, dim=0)
        anomaly_scores = (
            weights[0] * score_embedding +
            weights[1] * score_credibility +
            weights[2] * score_consistency +
            weights[3] * score_temporal
        )
        
        # 二值化
        anomaly_flags = (anomaly_scores > 0.5).float()
        
        return anomaly_scores, anomaly_flags


class Stage3Module(nn.Module):
    """
    Stage 3 完整模块
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_sensors: int = 7,
        output_dim: int = 4,
        sensor_to_output: Dict[int, int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.credibility_refiner = CredibilityRefinement(hidden_dim, dropout)
        
        self.aggregator = WeightedAggregation(
            hidden_dim, num_sensors, output_dim, sensor_to_output
        )
        
        self.uncertainty_estimator = UncertaintyEstimation(
            hidden_dim, output_dim, dropout
        )
        
        self.anomaly_detector = AnomalyDetector(hidden_dim, dropout=dropout)
        
        # 传感器输出投影（用于计算方差和一致性）
        self.sensor_output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        tau_temporal: torch.Tensor,
        raw_input: torch.Tensor,
        adj: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: (batch, N, T, H)
            tau_temporal: (batch, N, T)
            raw_input: (batch, N, T, F)
            adj: (N, N)
            
        Returns:
            dict with all outputs
        """
        B, N, T, H = h.shape
        
        # 1. 可信度精炼
        tau_final = self.credibility_refiner(h, tau_temporal)
        
        # 2. 加权聚合
        y_hat, fusion_weights = self.aggregator(h, tau_final)
        
        # 3. 计算传感器输出（用于不确定性）
        sensor_outputs = self.sensor_output_proj(h)  # (B, N, T, output_dim)
        
        # 4. 不确定性估计
        sigma = self.uncertainty_estimator(h, tau_final, y_hat, sensor_outputs)
        
        # 5. 邻居一致性
        neighbor_consistency = self._compute_neighbor_consistency(
            raw_input, adj
        )
        
        # 6. 异常检测
        anomaly_scores, anomaly_flags = self.anomaly_detector(
            h, tau_final, raw_input, neighbor_consistency
        )
        
        # 7. 系统整体可信度
        system_confidence = self._compute_system_confidence(
            tau_final, anomaly_flags
        )
        
        return {
            'y_hat': y_hat,              # (B, T, output_dim)
            'tau': tau_final,            # (B, N)
            'tau_full': tau_temporal,    # (B, N, T)
            'sigma': sigma,              # (B, T, output_dim)
            'anomaly_scores': anomaly_scores,  # (B, N)
            'anomaly_flags': anomaly_flags,    # (B, N)
            'fusion_weights': fusion_weights,  # (B, N, output_dim)
            'system_confidence': system_confidence,  # (B,)
        }
    
    def _compute_neighbor_consistency(
        self,
        raw_input: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """计算与邻居的一致性"""
        B, N, T, F = raw_input.shape
        
        # 简化：使用时间平均值
        x_mean = raw_input.mean(dim=2).squeeze(-1)  # (B, N)
        
        # 邻居平均
        neighbor_mean = torch.matmul(adj, x_mean.unsqueeze(-1)).squeeze(-1)  # (B, N)
        
        # 一致性 = 1 - 归一化差异
        diff = (x_mean - neighbor_mean).abs()
        diff_normalized = diff / (diff.max(dim=1, keepdim=True)[0] + 1e-8)
        consistency = 1 - diff_normalized
        
        return consistency
    
    def _compute_system_confidence(
        self,
        tau: torch.Tensor,
        anomaly_flags: torch.Tensor
    ) -> torch.Tensor:
        """
        计算系统整体可信度
        考虑：平均可信度、异常传感器比例
        """
        # 平均可信度
        avg_credibility = tau.mean(dim=1)  # (B,)
        
        # 非异常传感器比例
        healthy_ratio = 1 - anomaly_flags.mean(dim=1)  # (B,)
        
        # 综合系统置信度
        system_confidence = 0.6 * avg_credibility + 0.4 * healthy_ratio
        
        return system_confidence
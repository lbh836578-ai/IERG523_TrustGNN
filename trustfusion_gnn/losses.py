"""
损失函数
对应你的公式: L = ||Ŷ - Y*||² + λ₁L_consistency + λ₂L_credibility
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GroundTruth:
    """训练标签"""
    clean_data: torch.Tensor           # 干净数据 (B, N, T, F)
    fusion_target: torch.Tensor        # 融合目标 (B, T, output_F)
    fault_mask: torch.Tensor           # 故障掩码 (B, N, T)
    fault_types: torch.Tensor          # 故障类型 (B, N, T)
    credibility_target: torch.Tensor   # 可信度目标 (B, N, T)


class TrustFusionLoss(nn.Module):
    """
    复合损失函数
    L = L_fusion + λ₁·L_consistency + λ₂·L_credibility
    """
    
    def __init__(
        self,
        lambda_fusion: float = 1.0,
        lambda_consistency: float = 0.3,
        lambda_credibility: float = 0.5,
        lambda_anomaly: float = 0.3,
        lambda_uncertainty: float = 0.1
    ):
        super().__init__()
        
        self.lambda_fusion = lambda_fusion
        self.lambda_consistency = lambda_consistency
        self.lambda_credibility = lambda_credibility
        self.lambda_anomaly = lambda_anomaly
        self.lambda_uncertainty = lambda_uncertainty
        
    def forward(
        self,
        output,  # SystemOutput
        target: GroundTruth,
        raw_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        """
        losses = {}
        
        # 1. 融合损失 ||Ŷ - Y*||²
        loss_fusion = F.mse_loss(output.Y_hat, target.fusion_target)
        losses['fusion'] = loss_fusion.item()
        
        # 2. 可信度损失
        loss_credibility = F.binary_cross_entropy(
            output.tau_full,
            target.credibility_target,
            reduction='mean'
        )
        losses['credibility'] = loss_credibility.item()
        
        # 3. 异常检测损失
        # 以窗口级二值标签监督异常检测：窗口内出现过故障即记为异常
        anomaly_target = target.fault_mask.float().max(dim=-1).values  # (B, N)
        # 正样本加权，缓解异常样本相对稀疏导致的低召回
        pos_ratio = anomaly_target.mean()
        pos_weight = ((1.0 - pos_ratio) / (pos_ratio + 1e-6)).clamp(min=1.0, max=5.0)
        sample_weight = torch.where(
            anomaly_target > 0.5,
            torch.full_like(anomaly_target, pos_weight),
            torch.ones_like(anomaly_target)
        )
        loss_anomaly = F.binary_cross_entropy(
            output.anomaly_scores,
            anomaly_target,
            weight=sample_weight,
            reduction='mean'
        )
        losses['anomaly'] = loss_anomaly.item()
        
        # 4. 时空一致性损失
        loss_consistency = self._consistency_loss(
            output.tau, output.anomaly_scores
        )
        losses['consistency'] = loss_consistency.item()
        
        # 5. 不确定性校准损失
        loss_uncertainty = self._uncertainty_loss(
            output.Y_hat, target.fusion_target, output.sigma
        )
        losses['uncertainty'] = loss_uncertainty.item()
        
        # 总损失
        total_loss = (
            self.lambda_fusion * loss_fusion +
            self.lambda_credibility * loss_credibility +
            self.lambda_anomaly * loss_anomaly +
            self.lambda_consistency * loss_consistency +
            self.lambda_uncertainty * loss_uncertainty
        )
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def _consistency_loss(
        self,
        tau: torch.Tensor,
        anomaly_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        一致性损失：可信度和异常分数应该互补
        τ + anomaly ≈ 1
        """
        consistency = tau + anomaly_scores
        return ((consistency - 1.0) ** 2).mean()
    
    def _uncertainty_loss(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        不确定性校准损失
        负对数似然
        """
        sigma = sigma.clamp(min=1e-4)
        
        nll = 0.5 * (
            torch.log(2 * 3.14159 * sigma ** 2) +
            ((y_hat - y_true) ** 2) / (sigma ** 2)
        )
        
        return nll.mean()
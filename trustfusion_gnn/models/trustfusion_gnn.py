"""
TrustFusion-GNN 完整模型
整合 Stage 1, 2, 3
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .stage1_feature import Stage1Module
from .stage2_graph import Stage2Module
from .stage3_fusion import Stage3Module
from data_structures import SystemInput, SystemOutput


class TrustFusionGNN(nn.Module):
    """
    TrustFusion-GNN: 可信数据融合网络
    
    输入: X = {X, A, M}
    输出: Ŷ, τ, σ = f_θ(X)
    """
    
    def __init__(
        self,
        num_sensors: int = 7,
        input_dim: int = 1,
        output_dim: int = 4,
        hidden_dim: int = 64,
        temporal_layers: int = 2,
        gnn_layers: int = 2,
        num_heads: int = 4,
        statistical_features: int = 8,
        sensor_to_output: Dict[int, int] = None,
        dropout: float = 0.1,
        use_learnable_graph: bool = True
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.output_dim = output_dim
        
        # 默认传感器到输出的映射
        if sensor_to_output is None:
            # temp_1, temp_2 → 温度(0)
            # humidity_1, humidity_2 → 湿度(1)
            # soil_1, soil_2 → 土壤(2)
            # light → 光照(3)
            sensor_to_output = {
                0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 2, 6: 3
            }
        
        # Stage 1: 特征提取与初始可信度
        self.stage1 = Stage1Module(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            statistical_features=statistical_features,
            dropout=dropout
        )
        
        # Stage 2: 可信度感知GNN
        self.stage2 = Stage2Module(
            hidden_dim=hidden_dim,
            num_layers=gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_learnable_graph=use_learnable_graph
        )
        
        # Stage 3: 融合输出
        self.stage3 = Stage3Module(
            hidden_dim=hidden_dim,
            num_sensors=num_sensors,
            output_dim=output_dim,
            sensor_to_output=sensor_to_output,
            dropout=dropout
        )
        
    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        M: Optional[torch.Tensor] = None
    ) -> SystemOutput:
        """
        Args:
            X: (batch, N, T, F) 多传感器观测序列
            A: (N, N) 传感器空间关系图
            M: (N, D) 传感器元信息（可选）
            
        Returns:
            SystemOutput 包含所有输出
        """
        # Stage 1: 特征提取
        h_temp, s_feat, tau_init = self.stage1(X)
        
        # Stage 2: 图神经网络
        h_gnn, tau_temporal, learned_adj, attention = self.stage2(
            h_temp, tau_init, A
        )
        
        # Stage 3: 融合输出
        stage3_output = self.stage3(
            h_gnn, tau_temporal, X, learned_adj
        )
        
        # 构建输出
        return SystemOutput(
            Y_hat=stage3_output['y_hat'],
            tau=stage3_output['tau'],
            tau_full=stage3_output['tau_full'],
            anomaly_flags=stage3_output['anomaly_flags'],
            anomaly_scores=stage3_output['anomaly_scores'],
            sigma=stage3_output['sigma'],
            system_confidence=stage3_output['system_confidence'],
            node_embeddings=h_gnn,
            learned_adjacency=learned_adj,
            attention_weights=attention
        )
    
    def get_model_summary(self) -> Dict:
        """获取模型摘要"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # float32
            'num_sensors': self.num_sensors,
            'output_dim': self.output_dim
        }
"""
Stage 2: 可信度感知的图神经网络信息融合
对应你架构图中的：
- Credibility-Aware Graph Attention: α_ij = softmax(τ_i · τ_j · attention(h_i, h_j))
- Multi-hop Message Passing: h_i^{(l+1)} = Update(h_i^{(l)}, Σ α_ij · h_j^{(l)})
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class CredibilityAwareGraphAttention(nn.Module):
    """
    可信度感知的图注意力
    α_ij = softmax(τ_i · τ_j · attention(h_i, h_j))
    高可信度节点获得更高的注意力权重
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query, Key, Value 投影
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # 可信度调制
        self.credibility_proj = nn.Linear(1, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        adj_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, H) 节点特征
            tau: (batch, N) 可信度分数
            adj_mask: (N, N) 邻接矩阵掩码
            
        Returns:
            h_out: (batch, N, H) 更新后的节点特征
            attention: (batch, num_heads, N, N) 注意力权重
        """
        B, N, H = h.shape
        
        # 计算 Q, K, V
        Q = self.W_q(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, N, head_dim)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # (B, num_heads, N, N)
        
        # 可信度调制: τ_i · τ_j
        tau_i = tau.unsqueeze(-1)  # (B, N, 1)
        tau_j = tau.unsqueeze(-2)  # (B, 1, N)
        credibility_weight = tau_i * tau_j  # (B, N, N)
        
        # 投影到多头
        cred_bias = self.credibility_proj(credibility_weight.unsqueeze(-1))  # (B, N, N, num_heads)
        cred_bias = cred_bias.permute(0, 3, 1, 2)  # (B, num_heads, N, N)
        
        # 合并注意力分数
        attn_scores = attn_scores + cred_bias
        
        # 应用邻接掩码
        adj_mask_expanded = adj_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        attn_scores = attn_scores.masked_fill(adj_mask_expanded == 0, float('-inf'))
        
        # Softmax
        attention = F.softmax(attn_scores, dim=-1)
        attention = self.dropout(attention)
        
        # 聚合
        h_out = torch.matmul(attention, V)  # (B, num_heads, N, head_dim)
        h_out = h_out.transpose(1, 2).contiguous().view(B, N, H)
        h_out = self.output_proj(h_out)
        
        return h_out, attention.mean(dim=1)  # 返回平均注意力


class MultiHopMessagePassing(nn.Module):
    """
    多跳消息传递
    h_i^{(l+1)} = Update(h_i^{(l)}, Σ α_ij · h_j^{(l)})
    通过多跳传播让可信信息覆盖异常节点
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 可信度感知注意力
        self.attention = CredibilityAwareGraphAttention(
            hidden_dim, num_heads, dropout
        )
        
        # 更新函数
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        adj_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, H)
            tau: (batch, N)
            adj_mask: (N, N)
        Returns:
            h_out: (batch, N, H)
            attention: (batch, N, N)
        """
        # 注意力聚合
        h_attn, attention = self.attention(h, tau, adj_mask)
        
        # 更新: 结合自身和邻居信息
        h_combined = torch.cat([h, h_attn], dim=-1)
        h_update = self.update_mlp(h_combined)
        
        # 残差 + LayerNorm
        h = self.norm1(h + h_update)
        
        # FFN
        h = self.norm2(h + self.ffn(h))
        
        return h, attention


class Stage2Module(nn.Module):
    """
    Stage 2 完整模块
    多层可信度感知图神经网络
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_learnable_graph: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_learnable_graph = use_learnable_graph
        
        # 多层消息传递
        self.layers = nn.ModuleList([
            MultiHopMessagePassing(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 可学习图（如果启用）
        if use_learnable_graph:
            self.graph_learner = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # 可信度更新
        self.tau_updater = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        h: torch.Tensor,
        tau: torch.Tensor,
        adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, N, T, H) 时序节点特征
            tau: (batch, N) 初始可信度
            adj: (N, N) 基础邻接矩阵
            
        Returns:
            h_out: (batch, N, T, H) 更新后的节点特征
            tau_refined: (batch, N, T) 精炼后的时变可信度
            learned_adj: (N, N) 学习到的邻接矩阵
            attention_weights: (batch, N, N) 最终注意力权重
        """
        B, N, T, H = h.shape
        
        # 学习图结构
        learned_adj = adj
        if self.use_learnable_graph:
            # 使用时间平均的特征学习图
            h_mean = h.mean(dim=2)  # (B, N, H)
            h_proj = self.graph_learner(h_mean)  # (B, N, H)
            
            # 计算相似度
            h_norm = F.normalize(h_proj, dim=-1)
            sim = torch.bmm(h_norm, h_norm.transpose(1, 2))  # (B, N, N)
            sim = sim.mean(dim=0)  # (N, N) batch平均
            
            # 与基础图混合
            learned_adj = 0.7 * adj + 0.3 * torch.softmax(sim, dim=-1)
        
        # 存储时变可信度
        h_out_list = []
        tau_list = []
        attention_weights = None
        
        # 对每个时间步处理
        for t in range(T):
            h_t = h[:, :, t, :]  # (B, N, H)
            
            # 多层消息传递
            tau_t = tau
            for layer in self.layers:
                h_t, attention_weights = layer(h_t, tau_t, learned_adj)
                # 更新可信度
                tau_t = self.tau_updater(h_t).squeeze(-1)  # (B, N)
            
            h_out_list.append(h_t)
            tau_list.append(tau_t)
        
        # 堆叠时序输出特征
        h_out = torch.stack(h_out_list, dim=2)  # (B, N, T, H)

        # 堆叠时变可信度
        tau_refined = torch.stack(tau_list, dim=2)  # (B, N, T)
        
        return h_out, tau_refined, learned_adj, attention_weights
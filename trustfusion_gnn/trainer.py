"""
训练器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import copy

from config import SystemConfig
from models.trustfusion_gnn import TrustFusionGNN
from losses import TrustFusionLoss, GroundTruth
from metrics import MetricsCalculator, MetricsResult
from graph_builder import GraphBuilder
from data_structures import SystemOutput
from normalization import DataNormalizer


class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: TrustFusionGNN,
        config: SystemConfig,
        device: str = None
    ):
        self.model = model
        self.config = config
        self.device = device or config.device
        
        self.model.to(self.device)
        
        # 构建图
        self.graph_builder = GraphBuilder(config)
        self.adj = self.graph_builder.get_combined_adjacency().to(self.device)
        
        # 损失函数
        self.criterion = TrustFusionLoss(
            lambda_fusion=config.lambda_fusion,
            lambda_consistency=config.lambda_consistency,
            lambda_credibility=config.lambda_credibility,
            lambda_anomaly=0.6
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator(
            anomaly_threshold=config.anomaly_threshold
        )

        # 数据归一化
        self.normalizer = DataNormalizer(config)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_f1': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            X, fusion_target, fault_mask, credibility_target = batch
            
            # 移动到设备
            X = X.to(self.device)
            fusion_target = fusion_target.to(self.device)
            fault_mask = fault_mask.to(self.device)
            credibility_target = credibility_target.to(self.device)

            X_norm = self.normalizer.normalize_input(X)
            fusion_target_norm = self.normalizer.normalize_output(fusion_target)
            
            # 前向传播
            output = self.model(X_norm, self.adj)
            
            # 构造 GroundTruth 对象
            gt = GroundTruth(
                clean_data=X,  # 简化：用输入代替
                fusion_target=fusion_target_norm,
                fault_mask=fault_mask,
                fault_types=fault_mask,  # 简化
                credibility_target=credibility_target
            )
            
            # 计算损失
            loss, losses_dict = self.criterion(output, gt, X_norm)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累积损失
            for key, value in losses_dict.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def evaluate(
        self, 
        val_loader: DataLoader
    ) -> Tuple[Dict[str, float], MetricsResult]:
        """评估模型"""
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X, fusion_target, fault_mask, credibility_target = batch
                
                X = X.to(self.device)
                fusion_target = fusion_target.to(self.device)
                fault_mask = fault_mask.to(self.device)
                credibility_target = credibility_target.to(self.device)

                X_norm = self.normalizer.normalize_input(X)
                fusion_target_norm = self.normalizer.normalize_output(fusion_target)
                
                # 前向传播
                output = self.model(X_norm, self.adj)
                
                # 构造 GroundTruth
                gt = GroundTruth(
                    clean_data=X_norm,
                    fusion_target=fusion_target_norm,
                    fault_mask=fault_mask,
                    fault_types=fault_mask,
                    credibility_target=credibility_target
                )
                
                # 计算损失
                loss, losses_dict = self.criterion(output, gt, X_norm)

                y_hat_denorm = self.normalizer.denormalize_output(output.Y_hat)
                
                # 累积损失
                for key, value in losses_dict.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value
                num_batches += 1
                
                # 更新指标
                self.metrics_calculator.update(
                    y_hat=y_hat_denorm,
                    y_true=fusion_target,
                    anomaly_scores=output.anomaly_scores,
                    anomaly_labels=fault_mask.max(dim=-1).values,
                    tau=output.tau,
                    tau_target=credibility_target.mean(dim=-1),
                    system_confidence=output.system_confidence,
                    raw_input=None
                )
        
        # 平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # 计算指标
        metrics = self.metrics_calculator.compute()
        
        return avg_losses, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        patience: int = None
    ) -> Dict:
        """完整训练流程"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        if patience is None:
            patience = self.config.patience
            
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\n开始训练 (共 {num_epochs} epochs, 早停 patience={patience})")
        print(f"设备: {self.device}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses, metrics = self.evaluate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_losses['total'])
            
            # 记录历史
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['val_mae'].append(metrics.mae)
            self.history['val_f1'].append(metrics.anomaly_f1)
            
            epoch_time = time.time() - start_time
            
            # 打印进度
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_losses['total']:.4f} | "
                  f"Val Loss: {val_losses['total']:.4f} | "
                  f"MAE: {metrics.mae:.4f} | "
                  f"F1: {metrics.anomaly_f1:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # 早停检查
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"  ✓ 新最佳模型！")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发！已 {patience} epochs 无改善。")
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n已恢复最佳模型 (val_loss={best_val_loss:.4f})")
        
        return self.history
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"模型已从 {path} 加载")
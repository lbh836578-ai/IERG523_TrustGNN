"""
推理模块
实时推理和异常检测
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from config import SystemConfig
from models.trustfusion_gnn import TrustFusionGNN
from graph_builder import GraphBuilder
from data_structures import SystemOutput, FusionResult
from normalization import DataNormalizer


class InferenceEngine:
    """推理引擎"""
    
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
        self.model.eval()
        
        # 图
        self.graph_builder = GraphBuilder(config)
        self.adj = self.graph_builder.get_combined_adjacency().to(self.device)
        
        # 传感器信息
        self.sensor_ids = list(config.sensors.keys())
        self.output_names = ['temperature', 'humidity', 'soil_moisture', 'light']
        self.output_units = ['°C', '%RH', '%', 'lux']

        # 预处理/后处理
        self.normalizer = DataNormalizer(config)
        
        # 缓存窗口
        self.data_buffer = []
        self.window_size = config.window_size
        
    def process_single(
        self,
        sensor_readings: Dict[str, float],
        timestamp: datetime = None
    ) -> Optional[FusionResult]:
        """
        处理单个时间步的传感器读数
        
        Args:
            sensor_readings: {sensor_id: value}
            timestamp: 时间戳
            
        Returns:
            FusionResult 或 None（缓冲区未满）
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # 转换为向量
        reading_vec = np.zeros(len(self.sensor_ids))
        for i, sid in enumerate(self.sensor_ids):
            if sid in sensor_readings:
                reading_vec[i] = sensor_readings[sid]
            else:
                # 缺失值处理
                reading_vec[i] = np.nan
                
        # 添加到缓冲区
        self.data_buffer.append(reading_vec)
        
        # 保持窗口大小
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
            
        # 缓冲区未满
        if len(self.data_buffer) < self.window_size:
            return None
            
        # 构造输入张量
        X = np.array(self.data_buffer)  # (T, N)
        X = X.T[:, :, np.newaxis]       # (N, T, 1)
        
        # 处理 NaN
        X = np.nan_to_num(X, nan=0.0)
        
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # (1, N, T, 1)
        X_norm = self.normalizer.normalize_input(X_tensor)
        
        # 推理
        with torch.no_grad():
            output = self.model(X_norm, self.adj)
            
        # 转换为应用层结果
        return self._convert_to_fusion_result(output, timestamp)
    
    def process_window(
        self,
        X: np.ndarray,
        timestamp: datetime = None
    ) -> FusionResult:
        """
        处理完整窗口
        
        Args:
            X: (N, T) or (N, T, F) 传感器数据
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # 确保形状正确
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
            
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
        X_norm = self.normalizer.normalize_input(X_tensor)
        
        with torch.no_grad():
            output = self.model(X_norm, self.adj)
            
        return self._convert_to_fusion_result(output, timestamp)
    
    def _convert_to_fusion_result(
        self,
        output: SystemOutput,
        timestamp: datetime
    ) -> FusionResult:
        """转换为应用层结果"""
        
        # 融合值（取最后时刻）
        y_hat = self.normalizer.denormalize_output(output.Y_hat)
        sigma = self.normalizer.denormalize_uncertainty(output.sigma)

        fused_values = {}
        uncertainties = {}
        for i, name in enumerate(self.output_names):
            fused_values[name] = y_hat[0, -1, i].item()
            uncertainties[name] = sigma[0, -1, i].item()
        
        # 传感器可信度
        sensor_credibility = {}
        for i, sid in enumerate(self.sensor_ids):
            sensor_credibility[sid] = output.tau[0, i].item()
        
        # 异常标识
        anomaly_flags = {}
        for i, sid in enumerate(self.sensor_ids):
            anomaly_flags[sid] = output.anomaly_scores[0, i].item() > self.config.anomaly_threshold
        
        # 融合权重
        # 简化处理
        fusion_weights = {sid: output.tau[0, i].item() for i, sid in enumerate(self.sensor_ids)}
        
        # 系统置信度
        system_confidence = output.system_confidence[0].item()
        
        # 生成报警
        alerts = []
        recommendations = []
        
        # 检查异常传感器
        for sid, is_anomaly in anomaly_flags.items():
            if is_anomaly:
                cred = sensor_credibility[sid]
                alerts.append(f"传感器 {sid} 检测到异常 (可信度: {cred:.2f})")
                recommendations.append(f"建议检查传感器 {sid}")
        
        # 检查系统置信度
        if system_confidence < self.config.system_alert_threshold:
            alerts.append(f"系统整体置信度较低: {system_confidence:.2f}")
            recommendations.append("建议进行系统维护检查")
        
        # 检查数值异常
        for name, value in fused_values.items():
            idx = self.output_names.index(name)
            # 简单阈值检查
            if name == 'temperature' and (value < 0 or value > 50):
                alerts.append(f"温度值异常: {value:.1f}°C")
            elif name == 'humidity' and (value < 0 or value > 100):
                alerts.append(f"湿度值异常: {value:.1f}%")
        
        return FusionResult(
            timestamp=timestamp,
            fused_values=fused_values,
            uncertainties=uncertainties,
            sensor_credibility=sensor_credibility,
            anomaly_flags=anomaly_flags,
            fusion_weights=fusion_weights,
            system_confidence=system_confidence,
            alerts=alerts,
            recommendations=recommendations
        )
    
    def reset_buffer(self):
        """重置数据缓冲区"""
        self.data_buffer = []


class RobustnessEvaluator:
    """鲁棒性评估器"""
    
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
        self.model.eval()
        
        self.graph_builder = GraphBuilder(config)
        self.adj = self.graph_builder.get_combined_adjacency().to(self.device)
        self.normalizer = DataNormalizer(config)
        
    def evaluate_robustness(
        self,
        clean_data: torch.Tensor,
        fusion_target: torch.Tensor,
        anomaly_ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ) -> Dict[float, Dict[str, float]]:
        """
        评估不同异常比例下的性能
        
        Args:
            clean_data: (num_samples, N, T, F) 干净数据
            fusion_target: (num_samples, T, output_F) 融合目标
            anomaly_ratios: 异常比例列表
            
        Returns:
            {ratio: {mae, rmse, ...}}
        """
        from data_simulator import AgriculturalDataSimulator, SensorDataTensor
        
        results = {}
        
        for ratio in anomaly_ratios:
            print(f"评估异常比例: {ratio*100:.0f}%")
            
            maes = []
            
            for i in range(clean_data.shape[0]):
                # 创建数据对象
                data = SensorDataTensor(
                    X=clean_data[i],
                    timestamps=np.arange(clean_data.shape[2]),
                    sensor_ids=list(self.config.sensors.keys())
                )
                
                # 注入故障
                if ratio > 0:
                    simulator = AgriculturalDataSimulator(self.config, seed=i)
                    faulty_data, _, _ = simulator.inject_faults(data, fault_ratio=ratio)
                    X = faulty_data.X.unsqueeze(0).to(self.device)
                else:
                    X = data.X.unsqueeze(0).to(self.device)

                X_norm = self.normalizer.normalize_input(X)
                
                # 推理
                with torch.no_grad():
                    output = self.model(X_norm, self.adj)

                y_hat = self.normalizer.denormalize_output(output.Y_hat)
                
                # 计算 MAE
                mae = (y_hat[0] - fusion_target[i].to(self.device)).abs().mean().item()
                maes.append(mae)
            
            results[ratio] = {
                'mae': np.mean(maes),
                'mae_std': np.std(maes)
            }
            
            print(f"  MAE: {results[ratio]['mae']:.4f} ± {results[ratio]['mae_std']:.4f}")
        
        return results
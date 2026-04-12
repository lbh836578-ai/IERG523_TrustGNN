"""
Inference module
Real-time inference and anomaly detection
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
    """Inference engine"""
    
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
        
        # Graph
        self.graph_builder = GraphBuilder(config)
        self.adj = self.graph_builder.get_combined_adjacency().to(self.device)
        
        # Sensor metadata
        self.sensor_ids = list(config.sensors.keys())
        self.output_names = ['temperature', 'humidity', 'soil_moisture', 'light']
        self.output_units = ['°C', '%RH', '%', 'lux']

        # Pre/post processing
        self.normalizer = DataNormalizer(config)
        
        # Window buffer
        self.data_buffer = []
        self.window_size = config.window_size
        
    def process_single(
        self,
        sensor_readings: Dict[str, float],
        timestamp: datetime = None
    ) -> Optional[FusionResult]:
        """
        Process sensor readings for a single time step
        
        Args:
            sensor_readings: {sensor_id: value}
            timestamp: timestamp
            
        Returns:
            FusionResult or None (if the buffer is not full yet)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Convert to vector
        reading_vec = np.zeros(len(self.sensor_ids))
        for i, sid in enumerate(self.sensor_ids):
            if sid in sensor_readings:
                reading_vec[i] = sensor_readings[sid]
            else:
                # Missing-value handling
                reading_vec[i] = np.nan
                
        # Append to buffer
        self.data_buffer.append(reading_vec)
        
        # Keep fixed window size
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
            
        # Buffer not full yet
        if len(self.data_buffer) < self.window_size:
            return None
            
        # Build input tensor
        X = np.array(self.data_buffer)  # (T, N)
        X = X.T[:, :, np.newaxis]       # (N, T, 1)
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)  # (1, N, T, 1)
        X_norm = self.normalizer.normalize_input(X_tensor)
        
        # Inference
        with torch.no_grad():
            output = self.model(X_norm, self.adj)
            
        # Convert to application-layer result
        return self._convert_to_fusion_result(output, timestamp)
    
    def process_window(
        self,
        X: np.ndarray,
        timestamp: datetime = None
    ) -> FusionResult:
        """
        Process a full window
        
        Args:
            X: (N, T) or (N, T, F) sensor data
            timestamp: timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Ensure the shape is valid
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
        """Convert to application-layer result"""
        
        # Fused values (take the last time step)
        y_hat = self.normalizer.denormalize_output(output.Y_hat)
        sigma = self.normalizer.denormalize_uncertainty(output.sigma)

        fused_values = {}
        uncertainties = {}
        for i, name in enumerate(self.output_names):
            fused_values[name] = y_hat[0, -1, i].item()
            uncertainties[name] = sigma[0, -1, i].item()
        
        # Sensor trust scores
        sensor_credibility = {}
        for i, sid in enumerate(self.sensor_ids):
            sensor_credibility[sid] = output.tau[0, i].item()
        
        # Anomaly flags
        anomaly_flags = {}
        for i, sid in enumerate(self.sensor_ids):
            anomaly_flags[sid] = output.anomaly_scores[0, i].item() > self.config.anomaly_threshold
        
        # Fusion weights
        # Simplified handling
        fusion_weights = {sid: output.tau[0, i].item() for i, sid in enumerate(self.sensor_ids)}
        
        # System confidence
        system_confidence = output.system_confidence[0].item()
        
        # Generate alerts
        alerts = []
        recommendations = []
        
        # Check anomalous sensors
        for sid, is_anomaly in anomaly_flags.items():
            if is_anomaly:
                cred = sensor_credibility[sid]
                alerts.append(f"Sensor {sid} anomaly detected (trust score: {cred:.2f})")
                recommendations.append(f"Recommend checking sensor {sid}")
        
        # Check system confidence
        if system_confidence < self.config.system_alert_threshold:
            alerts.append(f"Low overall system confidence: {system_confidence:.2f}")
            recommendations.append("Recommend system maintenance check")
        
        # Check out-of-range values
        for name, value in fused_values.items():
            idx = self.output_names.index(name)
            # Simple threshold checks
            if name == 'temperature' and (value < 0 or value > 50):
                alerts.append(f"Abnormal temperature value: {value:.1f}°C")
            elif name == 'humidity' and (value < 0 or value > 100):
                alerts.append(f"Abnormal humidity value: {value:.1f}%")
        
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
        """Reset data buffer"""
        self.data_buffer = []


class RobustnessEvaluator:
    """Robustness evaluator"""
    
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
        Evaluate performance under different anomaly ratios
        
        Args:
            clean_data: (num_samples, N, T, F) clean data
            fusion_target: (num_samples, T, output_F) fusion target
            anomaly_ratios: anomaly ratio list
            
        Returns:
            {ratio: {mae, rmse, ...}}
        """
        from data_simulator import AgriculturalDataSimulator, SensorDataTensor
        
        results = {}
        
        for ratio in anomaly_ratios:
            print(f"Evaluating anomaly ratio: {ratio*100:.0f}%")
            
            maes = []
            
            for i in range(clean_data.shape[0]):
                # Create data object
                data = SensorDataTensor(
                    X=clean_data[i],
                    timestamps=np.arange(clean_data.shape[2]),
                    sensor_ids=list(self.config.sensors.keys())
                )
                
                # Inject faults
                if ratio > 0:
                    simulator = AgriculturalDataSimulator(self.config, seed=i)
                    faulty_data, _, _ = simulator.inject_faults(data, fault_ratio=ratio)
                    X = faulty_data.X.unsqueeze(0).to(self.device)
                else:
                    X = data.X.unsqueeze(0).to(self.device)

                X_norm = self.normalizer.normalize_input(X)
                
                # Inference
                with torch.no_grad():
                    output = self.model(X_norm, self.adj)

                y_hat = self.normalizer.denormalize_output(output.Y_hat)
                
                # Compute MAE
                mae = (y_hat[0] - fusion_target[i].to(self.device)).abs().mean().item()
                maes.append(mae)
            
            results[ratio] = {
                'mae': np.mean(maes),
                'mae_std': np.std(maes)
            }
            
            print(f"  MAE: {results[ratio]['mae']:.4f} ± {results[ratio]['mae_std']:.4f}")
        
        return results
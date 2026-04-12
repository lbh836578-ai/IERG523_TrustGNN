"""
Data simulator
Generate synthetic sensor data with injected faults
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import random

from config import SystemConfig, SensorMeta, SensorType


class FaultType(Enum):
    """Fault types"""
    NONE = 0
    STUCK_AT = 1          # Stuck at a fixed value
    DRIFT = 2             # Drift
    NOISE = 3             # Increased noise
    SPIKE = 4             # Spike
    BIAS = 5              # Bias
    MISSING = 6           # Missing data (NaN)
    RANDOM = 7            # Random values


@dataclass
class FaultInfo:
    """Fault metadata"""
    sensor_id: str
    sensor_idx: int
    fault_type: FaultType
    start_time: int
    end_time: int
    parameters: Dict


@dataclass
class SensorDataTensor:
    """Sensor data tensor"""
    X: torch.Tensor           # (N, T, F)
    timestamps: np.ndarray    # (T,)
    sensor_ids: List[str]


@dataclass
class GroundTruth:
    """Training labels"""
    clean_data: torch.Tensor           # (N, T, F) clean data
    fusion_target: torch.Tensor        # (T, output_F) fusion target
    fault_mask: torch.Tensor           # (N, T) fault mask, 1=fault
    fault_types: torch.Tensor          # (N, T) fault type encoding
    credibility_target: torch.Tensor   # (N, T) trust target, 1=trustworthy


class AgriculturalDataSimulator:
    """Agricultural data simulator"""
    
    def __init__(self, config: SystemConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        self.sensor_ids = list(config.sensors.keys())
        self.num_sensors = len(self.sensor_ids)
        
    def generate_clean_data(
        self, 
        num_samples: int = 1,
        window_size: int = None
    ) -> List[SensorDataTensor]:
        """Generate clean sensor data"""
        if window_size is None:
            window_size = self.config.window_size
            
        samples = []
        
        for _ in range(num_samples):
            # Generate base environment values (ground truth)
            base_temp = np.random.uniform(18, 30)          # base temperature
            base_humidity = np.random.uniform(40, 75)      # base humidity
            base_soil = np.random.uniform(25, 55)          # base soil moisture
            base_light = np.random.uniform(5000, 40000)    # base illumination
            
            # Time axis
            t = np.arange(window_size)
            
            # Generate temporal variation (daily cycle)
            hour_cycle = 2 * np.pi * t / window_size
            
            # Temperature: daily variation + noise
            temp_variation = 3 * np.sin(hour_cycle) + 0.5 * np.random.randn(window_size)
            
            # Humidity: negatively correlated with temperature
            humidity_variation = -2 * np.sin(hour_cycle) + 1.0 * np.random.randn(window_size)
            
            # Soil moisture: slow variation
            soil_variation = 0.5 * np.cumsum(np.random.randn(window_size) * 0.1)
            soil_variation = soil_variation - soil_variation.mean()
            
            # Illumination: strong daily pattern
            light_variation = 15000 * np.maximum(0, np.sin(hour_cycle)) + 500 * np.random.randn(window_size)
            
            # Generate data per sensor
            data = np.zeros((self.num_sensors, window_size, 1))
            
            for i, sensor_id in enumerate(self.sensor_ids):
                sensor = self.config.sensors[sensor_id]
                
                if sensor.sensor_type == SensorType.TEMPERATURE:
                    # Temperature sensor: base + temporal variation + sensor-specific noise
                    base = base_temp + temp_variation
                    noise = np.random.randn(window_size) * sensor.noise_std
                    # Add spatial variation
                    spatial_offset = np.random.uniform(-1, 1)
                    data[i, :, 0] = base + noise + spatial_offset
                    
                elif sensor.sensor_type == SensorType.HUMIDITY:
                    base = base_humidity + humidity_variation
                    noise = np.random.randn(window_size) * sensor.noise_std
                    spatial_offset = np.random.uniform(-2, 2)
                    data[i, :, 0] = np.clip(base + noise + spatial_offset, 0, 100)
                    
                elif sensor.sensor_type == SensorType.SOIL_MOISTURE:
                    base = base_soil + soil_variation
                    noise = np.random.randn(window_size) * sensor.noise_std
                    spatial_offset = np.random.uniform(-5, 5)
                    data[i, :, 0] = np.clip(base + noise + spatial_offset, 0, 100)
                    
                elif sensor.sensor_type == SensorType.LIGHT:
                    base = base_light + light_variation
                    noise = np.random.randn(window_size) * sensor.noise_std
                    data[i, :, 0] = np.clip(base + noise, 0, 65535)
            
            # Create timestamps
            timestamps = np.arange(window_size)
            
            samples.append(SensorDataTensor(
                X=torch.FloatTensor(data),
                timestamps=timestamps,
                sensor_ids=self.sensor_ids.copy()
            ))
            
        return samples
    
    def inject_faults(
        self,
        data: SensorDataTensor,
        fault_ratio: float = 0.3,
        fault_duration_ratio: float = 0.5
    ) -> Tuple[SensorDataTensor, List[FaultInfo], torch.Tensor]:
        """
        Inject faults
        
        Args:
            data: clean data
            fault_ratio: ratio of faulty sensors
            fault_duration_ratio: ratio of fault duration
            
        Returns:
            faulty_data: data after fault injection
            fault_infos: fault metadata list
            fault_mask: fault mask (N, T)
        """
        X = data.X.clone()
        N, T, F = X.shape
        
        fault_mask = torch.zeros(N, T)
        fault_infos = []
        
        # Randomly choose faulty sensors
        num_faulty = max(1, int(N * fault_ratio))
        faulty_indices = random.sample(range(N), num_faulty)
        
        # Available fault types
        fault_types = [
            FaultType.STUCK_AT,
            FaultType.DRIFT,
            FaultType.NOISE,
            FaultType.SPIKE,
            FaultType.BIAS,
            FaultType.RANDOM
        ]
        
        for idx in faulty_indices:
            sensor_id = self.sensor_ids[idx]
            sensor = self.config.sensors[sensor_id]
            
            # Randomly choose fault type
            fault_type = random.choice(fault_types)
            
            # Randomly choose fault time range
            fault_duration = int(T * fault_duration_ratio * random.uniform(0.5, 1.0))
            start_time = random.randint(0, T - fault_duration)
            end_time = start_time + fault_duration
            
            # Inject fault
            params = {}
            
            if fault_type == FaultType.STUCK_AT:
                # Stuck-at fault
                stuck_value = X[idx, start_time, 0].item()
                X[idx, start_time:end_time, 0] = stuck_value
                params['stuck_value'] = stuck_value
                
            elif fault_type == FaultType.DRIFT:
                # Drift fault
                drift_rate = random.uniform(0.1, 0.5) * random.choice([-1, 1])
                drift = torch.arange(end_time - start_time).float() * drift_rate
                X[idx, start_time:end_time, 0] += drift
                params['drift_rate'] = drift_rate
                
            elif fault_type == FaultType.NOISE:
                # Increased noise fault
                noise_multiplier = random.uniform(3, 10)
                extra_noise = torch.randn(end_time - start_time) * sensor.noise_std * noise_multiplier
                X[idx, start_time:end_time, 0] += extra_noise
                params['noise_multiplier'] = noise_multiplier
                
            elif fault_type == FaultType.SPIKE:
                # Spike fault
                num_spikes = random.randint(3, 10)
                spike_times = random.sample(range(start_time, end_time), min(num_spikes, end_time - start_time))
                for st in spike_times:
                    spike_magnitude = random.uniform(5, 20) * random.choice([-1, 1])
                    X[idx, st, 0] += spike_magnitude
                params['num_spikes'] = num_spikes
                
            elif fault_type == FaultType.BIAS:
                # Bias fault
                bias = random.uniform(5, 15) * random.choice([-1, 1])
                X[idx, start_time:end_time, 0] += bias
                params['bias'] = bias
                
            elif fault_type == FaultType.RANDOM:
                # Random-value fault
                random_values = torch.rand(end_time - start_time) * (sensor.max_value - sensor.min_value) + sensor.min_value
                X[idx, start_time:end_time, 0] = random_values
                
            # Record fault mask
            fault_mask[idx, start_time:end_time] = 1.0
            
            # Record fault metadata
            fault_infos.append(FaultInfo(
                sensor_id=sensor_id,
                sensor_idx=idx,
                fault_type=fault_type,
                start_time=start_time,
                end_time=end_time,
                parameters=params
            ))
        
        faulty_data = SensorDataTensor(
            X=X,
            timestamps=data.timestamps,
            sensor_ids=data.sensor_ids
        )
        
        return faulty_data, fault_infos, fault_mask
    
    def compute_ground_truth(
        self,
        clean_data: SensorDataTensor,
        fault_mask: torch.Tensor
    ) -> GroundTruth:
        """Compute ground-truth labels"""
        N, T, F = clean_data.X.shape
        
        # Fusion target: average by sensor type group
        fusion_target = torch.zeros(T, 4)  # 4 output channels
        
        for i, sensor_id in enumerate(self.sensor_ids):
            sensor = self.config.sensors[sensor_id]
            group = sensor.fusion_group
            
            # Add to corresponding channel
            fusion_target[:, group] += clean_data.X[i, :, 0]
        
        # Normalize by sensor count per type
        group_counts = [0, 0, 0, 0]
        for sensor_id in self.sensor_ids:
            sensor = self.config.sensors[sensor_id]
            group_counts[sensor.fusion_group] += 1
            
        for g in range(4):
            if group_counts[g] > 0:
                fusion_target[:, g] /= group_counts[g]
        
        # Trust target: no fault=1, fault=0
        credibility_target = 1.0 - fault_mask
        
        return GroundTruth(
            clean_data=clean_data.X,
            fusion_target=fusion_target,
            fault_mask=fault_mask,
            fault_types=fault_mask,
            credibility_target=credibility_target
        )
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        inject_faults: bool = True,
        fault_ratio: float = 0.3
    ) -> Tuple[List[SensorDataTensor], List[GroundTruth], List[List[FaultInfo]]]:
        """
        Generate full dataset
        
        Returns:
            data_list: sensor data list (may include faults)
            gt_list: ground-truth label list
            fault_list: fault metadata list
        """
        # Generate clean data
        clean_samples = self.generate_clean_data(num_samples)
        
        data_list = []
        gt_list = []
        fault_list = []
        
        for clean_data in clean_samples:
            if inject_faults and random.random() < 0.8:  # 80% samples include faults
                faulty_data, faults, fault_mask = self.inject_faults(
                    clean_data, fault_ratio=fault_ratio
                )
                gt = self.compute_ground_truth(clean_data, fault_mask)
                data_list.append(faulty_data)
                fault_list.append(faults)
            else:
                fault_mask = torch.zeros(clean_data.X.shape[0], clean_data.X.shape[1])
                gt = self.compute_ground_truth(clean_data, fault_mask)
                data_list.append(clean_data)
                fault_list.append([])
                
            gt_list.append(gt)
            
        return data_list, gt_list, fault_list
    
    def create_dataloader(
        self,
        data_list: List[SensorDataTensor],
        gt_list: List[GroundTruth],
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader"""
        dataset = SensorDataset(data_list, gt_list)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
    
    @staticmethod
    def collate_fn(batch):
        """Batch collation function"""
        X = torch.stack([item[0] for item in batch])
        fusion_target = torch.stack([item[1] for item in batch])
        fault_mask = torch.stack([item[2] for item in batch])
        credibility_target = torch.stack([item[3] for item in batch])
        
        return X, fusion_target, fault_mask, credibility_target


class SensorDataset(Dataset):
    """Sensor dataset"""
    
    def __init__(
        self, 
        data_list: List[SensorDataTensor], 
        gt_list: List[GroundTruth]
    ):
        self.data_list = data_list
        self.gt_list = gt_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        gt = self.gt_list[idx]
        
        return (
            data.X,                    # (N, T, F)
            gt.fusion_target,          # (T, output_F)
            gt.fault_mask,             # (N, T)
            gt.credibility_target      # (N, T)
        )
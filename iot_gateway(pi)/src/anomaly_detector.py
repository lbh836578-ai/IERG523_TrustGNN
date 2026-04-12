# src/anomaly_detector.py
"""
异常检测模块
在边缘端进行初步的异常检测
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    边缘端异常检测器
    
    使用多种方法检测传感器异常:
    1. 范围检测 - 值超出有效范围
    2. 统计检测 - Z-score异常
    3. 突变检测 - 短时间内剧烈变化
    4. 一致性检测 - 与同类传感器比较
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化异常检测器
        
        参数:
            config: 配置字典
        """
        self.config = config or {}
        
        # 默认有效范围
        self.valid_ranges = {
            "temperature": (-40, 80),
            "humidity": (0, 100),
            "soil_moisture": (0, 100),
            "light": (0, 100000)
        }
        
        # 历史数据缓冲
        self.history: Dict[str, deque] = {}
        self.window_size = self.config.get("window_size", 20)
        
        # Z-score阈值
        self.z_threshold = self.config.get("z_threshold", 3.0)
        
        # 突变阈值(相对于标准差的倍数)
        self.spike_threshold = self.config.get("spike_threshold", 2.0)
        
        # 异常计数(用于检测持续异常)
        self.anomaly_counts: Dict[str, int] = {}
    
    def detect(
        self, 
        node_id: str, 
        sensor_type: str, 
        value: float
    ) -> Tuple[bool, List[str]]:
        """
        检测单个数据点是否异常
        
        参数:
            node_id: 节点ID
            sensor_type: 传感器类型
            value: 传感器值
            
        返回:
            (是否异常, 异常原因列表)
        """
        anomalies = []
        key = f"{node_id}_{sensor_type}"
        
        # 1. 范围检测
        if sensor_type in self.valid_ranges:
            min_val, max_val = self.valid_ranges[sensor_type]
            if value < min_val or value > max_val:
                anomalies.append(f"out_of_range: {value} not in [{min_val}, {max_val}]")
        
        # 获取历史数据
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window_size)
        
        history = list(self.history[key])
        
        # 2. 统计异常检测 (需要足够的历史数据)
        if len(history) >= 5:
            mean = np.mean(history)
            std = np.std(history)
            
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > self.z_threshold:
                    anomalies.append(f"statistical: z_score={z_score:.2f}")
        
        # 3. 突变检测
        if len(history) >= 1:
            last_value = history[-1]
            change = abs(value - last_value)
            
            # 计算允许的最大变化
            if len(history) >= 3:
                typical_change = np.std(np.diff(history)) if len(history) > 1 else 0
                if typical_change > 0 and change > self.spike_threshold * typical_change:
                    anomalies.append(f"spike: change={change:.2f}")
            else:
                # 使用默认阈值
                default_thresholds = {
                    "temperature": 5.0,
                    "humidity": 20.0,
                    "soil_moisture": 30.0,
                    "light": 10000.0
                }
                threshold = default_thresholds.get(sensor_type, float("inf"))
                if change > threshold:
                    anomalies.append(f"spike: change={change:.2f}")
        
        # 更新历史数据
        self.history[key].append(value)
        
        # 更新异常计数
        if anomalies:
            self.anomaly_counts[key] = self.anomaly_counts.get(key, 0) + 1
        else:
            self.anomaly_counts[key] = 0
        
        return len(anomalies) > 0, anomalies
    
    def detect_cross_sensor(
        self, 
        sensor_type: str, 
        values: Dict[str, float]
    ) -> Dict[str, Tuple[bool, List[str]]]:
        """
        跨传感器一致性检测
        
        参数:
            sensor_type: 传感器类型
            values: {node_id: value} 字典
            
        返回:
            {node_id: (是否异常, 异常原因)} 字典
        """
        if len(values) < 2:
            return {}
        
        results = {}
        all_values = list(values.values())
        median = np.median(all_values)
        mad = np.median([abs(v - median) for v in all_values])  # 中位数绝对偏差
        
        if mad == 0:
            mad = np.std(all_values) / 1.4826  # 使用标准差估计
        
        if mad == 0:
            return {}
        
        for node_id, value in values.items():
            # 计算修正Z分数
            modified_z = 0.6745 * (value - median) / mad
            
            if abs(modified_z) > self.z_threshold:
                results[node_id] = (
                    True, 
                    [f"inconsistent: modified_z={modified_z:.2f}, median={median:.2f}"]
                )
            else:
                results[node_id] = (False, [])
        
        return results
    
    def is_sensor_faulty(self, node_id: str, sensor_type: str) -> bool:
        """
        判断传感器是否可能故障
        
        连续多次异常表示可能故障
        """
        key = f"{node_id}_{sensor_type}"
        return self.anomaly_counts.get(key, 0) >= 5
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取所有传感器的健康状态"""
        status = {}
        
        for key, count in self.anomaly_counts.items():
            node_id, sensor_type = key.rsplit("_", 1)
            
            if node_id not in status:
                status[node_id] = {}
            
            status[node_id][sensor_type] = {
                "consecutive_anomalies": count,
                "possibly_faulty": count >= 5,
                "history_size": len(self.history.get(key, []))
            }
        
        return status
# src/data_processor.py
"""
数据处理模块
负责数据清洗、格式化和初步分析
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class SensorDataBuffer:
    """
    传感器数据缓冲区
    为每个传感器维护一个时间窗口的数据
    """
    
    def __init__(self, window_size: int = 20):
        """
        初始化缓冲区
        
        参数:
            window_size: 时间窗口大小(保留最近N个数据点)
        """
        self.window_size = window_size
        self.buffers: Dict[str, Dict[str, deque]] = {}
    
    def add_data(self, node_id: str, sensor_type: str, value: float):
        """添加数据点"""
        # 初始化节点缓冲区
        if node_id not in self.buffers:
            self.buffers[node_id] = {}
        
        # 初始化传感器缓冲区
        if sensor_type not in self.buffers[node_id]:
            self.buffers[node_id][sensor_type] = deque(maxlen=self.window_size)
        
        # 添加数据
        self.buffers[node_id][sensor_type].append(value)
    
    def get_history(self, node_id: str, sensor_type: str) -> List[float]:
        """获取历史数据"""
        if node_id in self.buffers and sensor_type in self.buffers[node_id]:
            return list(self.buffers[node_id][sensor_type])
        return []
    
    def get_statistics(self, node_id: str, sensor_type: str) -> Dict[str, float]:
        """计算统计信息"""
        history = self.get_history(node_id, sensor_type)
        
        if len(history) < 2:
            return {
                "mean": history[0] if history else 0,
                "std": 0,
                "min": history[0] if history else 0,
                "max": history[0] if history else 0,
                "count": len(history)
            }
        
        arr = np.array(history)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(history)
        }


class DataProcessor:
    """
    数据处理器
    
    功能:
    1. 数据验证和清洗
    2. 单位转换和标准化
    3. 计算数据质量分数
    4. 格式化为云端所需格式
    """
    
    # 传感器有效范围定义
    VALID_RANGES = {
        "temperature": {"min": -40, "max": 80, "unit": "celsius"},
        "humidity": {"min": 0, "max": 100, "unit": "percent"},
        "soil_moisture": {"min": 0, "max": 100, "unit": "percent"},
        "light": {"min": 0, "max": 100000, "unit": "lux"}
    }
    
    def __init__(self, window_size: int = 20):
        """
        初始化数据处理器
        
        参数:
            window_size: 时间窗口大小
        """
        self.buffer = SensorDataBuffer(window_size)
        self.processed_count = 0
    
    def process(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理原始传感器数据
        
        参数:
            raw_data: 从MQTT接收的原始数据
            
        返回:
            处理后的数据字典，如果数据无效则返回None
        """
        try:
            # 1. 提取基本信息
            node_id = raw_data.get("node_id", "unknown")
            timestamp = raw_data.get("timestamp", 0)
            device_quality = raw_data.get("quality", 1.0)
            
            # 2. 处理每个传感器的数据
            sensors_data = raw_data.get("sensors", {})
            processed_sensors = {}
            
            for sensor_type, sensor_info in sensors_data.items():
                processed = self._process_sensor(
                    node_id, 
                    sensor_type, 
                    sensor_info
                )
                if processed:
                    processed_sensors[sensor_type] = processed
            
            # 3. 如果没有有效数据，返回None
            if not processed_sensors:
                logger.warning(f"No valid sensor data from {node_id}")
                return None
            
            # 4. 构建输出
            result = {
                "node_id": node_id,
                "timestamp": datetime.now().isoformat(),
                "device_timestamp": timestamp,
                "device_quality": device_quality,
                "sensors": processed_sensors,
                "metadata": {
                    "gateway_id": "raspberry_pi_gateway",
                    "processing_version": "1.0"
                }
            }
            
            # 5. 添加设备状态信息
            if "device_status" in raw_data:
                result["device_status"] = raw_data["device_status"]
            
            self.processed_count += 1
            logger.debug(f"Processed data from {node_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def _process_sensor(
        self, 
        node_id: str, 
        sensor_type: str, 
        sensor_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """处理单个传感器数据"""
        
        # 检查数据有效性标志
        if not sensor_info.get("valid", True):
            return None
        
        value = sensor_info.get("value")
        
        # 检查值是否存在
        if value is None:
            return None
        
        # 获取有效范围
        valid_range = self.VALID_RANGES.get(sensor_type, {})
        min_val = valid_range.get("min", float("-inf"))
        max_val = valid_range.get("max", float("inf"))
        
        # 1. 范围检查
        in_range = min_val <= value <= max_val
        
        # 2. 添加到缓冲区
        if in_range:
            self.buffer.add_data(node_id, sensor_type, value)
        
        # 3. 获取统计信息
        stats = self.buffer.get_statistics(node_id, sensor_type)
        
        # 4. 计算数据质量分数
        quality = self._calculate_quality(value, stats, in_range)
        
        # 5. 检测异常
        is_anomaly = self._detect_anomaly(value, stats)
        
        return {
            "value": value,
            "unit": valid_range.get("unit", "unknown"),
            "quality": quality,
            "in_range": in_range,
            "is_anomaly": is_anomaly,
            "statistics": stats
        }
    
    def _calculate_quality(
        self, 
        value: float, 
        stats: Dict[str, float], 
        in_range: bool
    ) -> float:
        """
        计算数据质量分数
        
        质量分数基于:
        1. 是否在有效范围内
        2. 与历史均值的偏差
        3. 数据的稳定性(标准差)
        """
        quality = 1.0
        
        # 1. 范围检查
        if not in_range:
            quality *= 0.3
        
        # 2. 偏差检查
        if stats["count"] > 1 and stats["std"] > 0:
            z_score = abs(value - stats["mean"]) / stats["std"]
            
            if z_score > 3:
                quality *= 0.5
            elif z_score > 2:
                quality *= 0.7
            elif z_score > 1:
                quality *= 0.9
        
        return round(quality, 3)
    
    def _detect_anomaly(
        self, 
        value: float, 
        stats: Dict[str, float], 
        threshold: float = 3.0
    ) -> bool:
        """
        检测异常值
        
        使用Z-score方法: |值 - 均值| > threshold * 标准差
        """
        if stats["count"] < 5 or stats["std"] == 0:
            return False
        
        z_score = abs(value - stats["mean"]) / stats["std"]
        return z_score > threshold
    
    def get_formatted_batch(self, data_list: List[Dict]) -> Dict[str, Any]:
        """
        将多条数据格式化为云端批量上传格式
        
        参数:
            data_list: 处理后的数据列表
            
        返回:
            云端API所需的批量数据格式
        """
        return {
            "batch_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "gateway_id": "raspberry_pi_gateway",
            "timestamp": datetime.now().isoformat(),
            "data_count": len(data_list),
            "data": data_list
        }
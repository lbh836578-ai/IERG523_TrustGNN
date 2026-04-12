"""
Configuration Files - TrustFusion GNN System Configuration
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import torch


class SensorType(Enum):
    """传感器类型"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    SOIL_MOISTURE = "soil_moisture"
    LIGHT = "light"


@dataclass
class SensorMeta:
    """
    传感器元信息 M ∈ ℝ^{N×D}
    对应你架构中的：位置坐标、设备型号、安装时间等
    """
    sensor_id: str
    sensor_type: SensorType
    position: Tuple[float, float, float]  # (x, y, z) 三维坐标
    esp32_id: int                          # 所属ESP32
    install_date: str                      # 安装日期
    model: str                             # 设备型号
    
    # 物理属性
    unit: str
    min_value: float
    max_value: float
    normal_range: Tuple[float, float]
    accuracy: float                        # 标称精度
    noise_std: float                       # 正常噪声标准差
    
    # 输出所属的融合组（哪些传感器融合到同一输出通道）
    fusion_group: int                      # 0=温度, 1=湿度, 2=土壤, 3=光照


@dataclass
class SystemConfig:
    """系统总配置"""
    
    # ========== 数据维度参数（对应你的 N, T, F）==========
    num_sensors: int = 7                   # N: 传感器数量
    window_size: int = 60                  # T: 时间窗口长度
    input_features: int = 1                # F: 每个传感器的输入特征维度
    output_features: int = 4               # 输出特征维度（温度、湿度、土壤、光照）
    
    # ========== 采样参数 ==========
    sampling_rate_hz: float = 1.0          # 采样频率
    evaluation_interval: int = 5           # 评估间隔（秒）
    
    # ========== 元信息维度 D ==========
    meta_dim: int = 8                      # 传感器元信息嵌入维度
    
    # ========== 模型架构参数 ==========
    # Stage 1: 特征提取
    temporal_hidden_dim: int = 64
    temporal_layers: int = 2
    statistical_features: int = 8          # 统计特征数量
    
    # Stage 2: 图神经网络
    gnn_hidden_dim: int = 64
    gnn_layers: int = 2
    num_attention_heads: int = 4
    
    # Stage 3: 融合输出
    fusion_hidden_dim: int = 64
    
    # 通用
    dropout: float = 0.1
    
    # ========== 图构建参数 ==========
    spatial_k_neighbors: int = 3
    correlation_threshold: float = 0.5
    use_learnable_graph: bool = True
    
    # ========== 损失函数权重（对应你的 λ）==========
    lambda_fusion: float = 1.0             # λ_fusion: 融合损失
    lambda_consistency: float = 0.3        # λ_consistency: 时空一致性
    lambda_credibility: float = 0.5        # λ_credibility: 可信度约束
    
    # ========== 异常检测阈值 ==========
    anomaly_threshold: float = 0.4         # 异常分数阈值
    credibility_low_threshold: float = 0.3 # 低可信度阈值
    credibility_high_threshold: float = 0.7
    system_alert_threshold: float = 0.5    # 系统整体可信度报警阈值
    
    # ========== 训练参数 ==========
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 15
    
    # ========== 设备 ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========== 传感器配置 ==========
    sensors: Dict[str, SensorMeta] = field(default_factory=dict)
    
    # ========== 物理约束（用于一致性损失）==========
    physical_constraints: Dict = field(default_factory=dict)


def get_agricultural_config() -> SystemConfig:
    """
    获取农业监控配置
    你的硬件：2×温湿度传感器, 2×土壤湿度传感器, 1×光照传感器, 3×ESP32
    """
    config = SystemConfig()
    
    # 定义7个传感器
    config.sensors = {
        # ===== ESP32 #1: 温湿度传感器1 (DHT22) =====
        "temp_1": SensorMeta(
            sensor_id="temp_1",
            sensor_type=SensorType.TEMPERATURE,
            position=(0.0, 0.0, 0.5),
            esp32_id=1,
            install_date="2024-01-01",
            model="DHT22",
            unit="°C",
            min_value=-40.0,
            max_value=80.0,
            normal_range=(15.0, 35.0),
            accuracy=0.5,
            noise_std=0.3,
            fusion_group=0  # 融合到温度通道
        ),
        "humidity_1": SensorMeta(
            sensor_id="humidity_1",
            sensor_type=SensorType.HUMIDITY,
            position=(0.0, 0.0, 0.5),
            esp32_id=1,
            install_date="2024-01-01",
            model="DHT22",
            unit="%RH",
            min_value=0.0,
            max_value=100.0,
            normal_range=(30.0, 80.0),
            accuracy=2.0,
            noise_std=1.5,
            fusion_group=1  # 融合到湿度通道
        ),
        
        # ===== ESP32 #2: 温湿度传感器2 (DHT22) =====
        "temp_2": SensorMeta(
            sensor_id="temp_2",
            sensor_type=SensorType.TEMPERATURE,
            position=(5.0, 0.0, 0.5),
            esp32_id=2,
            install_date="2024-01-01",
            model="DHT22",
            unit="°C",
            min_value=-40.0,
            max_value=80.0,
            normal_range=(15.0, 35.0),
            accuracy=0.5,
            noise_std=0.3,
            fusion_group=0
        ),
        "humidity_2": SensorMeta(
            sensor_id="humidity_2",
            sensor_type=SensorType.HUMIDITY,
            position=(5.0, 0.0, 0.5),
            esp32_id=2,
            install_date="2024-01-01",
            model="DHT22",
            unit="%RH",
            min_value=0.0,
            max_value=100.0,
            normal_range=(30.0, 80.0),
            accuracy=2.0,
            noise_std=1.5,
            fusion_group=1
        ),
        
        # ===== ESP32 #3: 土壤湿度×2 + 光照 =====
        "soil_1": SensorMeta(
            sensor_id="soil_1",
            sensor_type=SensorType.SOIL_MOISTURE,
            position=(2.0, 3.0, -0.2),  # 地下20cm
            esp32_id=3,
            install_date="2024-01-01",
            model="Capacitive_v1.2",
            unit="%",
            min_value=0.0,
            max_value=100.0,
            normal_range=(20.0, 60.0),
            accuracy=3.0,
            noise_std=2.0,
            fusion_group=2  # 融合到土壤湿度通道
        ),
        "soil_2": SensorMeta(
            sensor_id="soil_2",
            sensor_type=SensorType.SOIL_MOISTURE,
            position=(3.0, 3.0, -0.2),
            esp32_id=3,
            install_date="2024-01-01",
            model="Capacitive_v1.2",
            unit="%",
            min_value=0.0,
            max_value=100.0,
            normal_range=(20.0, 60.0),
            accuracy=3.0,
            noise_std=2.0,
            fusion_group=2
        ),
        "light": SensorMeta(
            sensor_id="light",
            sensor_type=SensorType.LIGHT,
            position=(2.5, 1.5, 1.0),
            esp32_id=3,
            install_date="2024-01-01",
            model="BH1750",
            unit="lux",
            min_value=0.0,
            max_value=65535.0,
            normal_range=(100.0, 50000.0),
            accuracy=1.0,
            noise_std=200.0,
            fusion_group=3  # 融合到光照通道
        ),
    }
    
    config.num_sensors = len(config.sensors)
    
    # 物理约束关系
    config.physical_constraints = {
        # 温度-湿度负相关
        ("temp_1", "humidity_1"): {"type": "negative_corr", "strength": 0.5},
        ("temp_2", "humidity_2"): {"type": "negative_corr", "strength": 0.5},
        # 同类型传感器应接近
        ("temp_1", "temp_2"): {"type": "similar", "strength": 0.8, "max_diff": 3.0},
        ("humidity_1", "humidity_2"): {"type": "similar", "strength": 0.7, "max_diff": 10.0},
        ("soil_1", "soil_2"): {"type": "similar", "strength": 0.6, "max_diff": 15.0},
        # 温度与光照白天正相关
        ("temp_1", "light"): {"type": "positive_corr", "strength": 0.4, "condition": "daytime"},
    }
    
    return config
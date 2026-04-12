#!/usr/bin/env python3
# main.py
"""
TrustFusion-GNN 树莓派网关主程序

功能:
1. 接收ESP32传感器节点的MQTT数据
2. 数据处理和初步异常检测
3. 本地存储
4. 上传到云端
"""

import os
import sys
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Any

import yaml

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mqtt_handler import MQTTHandler
from data_processor import DataProcessor
from local_storage import LocalStorage
from cloud_uploader import CloudUploader
from anomaly_detector import AnomalyDetector


class Gateway:
    """
    IoT网关主类
    
    整合所有模块，协调数据流转
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化网关
        
        参数:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 设置日志
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing IoT Gateway...")
        
        # 初始化各模块
        self._init_modules()
        
        # 运行状态
        self.running = False
        
        # 统计信息
        self.stats = {
            "start_time": None,
            "messages_received": 0,
            "messages_processed": 0,
            "anomalies_detected": 0
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"Config file not found: {config_path}")
            print("Creating default config...")
            self._create_default_config(config_path)
        
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self, config_path: str):
        """创建默认配置文件"""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        default_config = {
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "username": "",
                "password": "",
                "topics": {
                    "sensor_data": "farm/sensors",
                    "control": "farm/control/#",
                    "status": "farm/status/#"
                }
            },
            "cloud": {
                "enabled": False,
                "api_url": "http://your-cloud-server.com/api",
                "api_key": "your-api-key",
                "upload_interval": 30
            },
            "processing": {
                "window_size": 20,
                "anomaly_threshold": 3.0
            },
            "storage": {
                "database_path": "data/sensor_data.db",
                "max_records": 100000
            },
            "logging": {
                "level": "INFO",
                "file": "logs/gateway.log"
            }
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_file = log_config.get("file", "logs/gateway.log")
        
        # 确保日志目录存在
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
    
    def _init_modules(self):
        """初始化各功能模块"""
        mqtt_config = self.config.get("mqtt", {})
        cloud_config = self.config.get("cloud", {})
        storage_config = self.config.get("storage", {})
        processing_config = self.config.get("processing", {})
        
        # MQTT处理器
        self.mqtt_handler = MQTTHandler(
            broker=mqtt_config.get("broker", "localhost"),
            port=mqtt_config.get("port", 1883),
            username=mqtt_config.get("username", ""),
            password=mqtt_config.get("password", "")
        )
        
        # 数据处理器
        self.data_processor = DataProcessor(
            window_size=processing_config.get("window_size", 20)
        )
        
        # 本地存储
        self.local_storage = LocalStorage(
            db_path=storage_config.get("database_path", "data/sensor_data.db"),
            max_records=storage_config.get("max_records", 100000)
        )
        
        # 异常检测器
        self.anomaly_detector = AnomalyDetector(processing_config)
        
        # 云端上传器
        if cloud_config.get("enabled", False):
            self.cloud_uploader = CloudUploader(
                api_url=cloud_config.get("api_url", ""),
                api_key=cloud_config.get("api_key", ""),
                upload_interval=cloud_config.get("upload_interval", 30)
            )
        else:
            self.cloud_uploader = None
            self.logger.info("Cloud upload disabled")
        
        self.logger.info("All modules initialized")
    
    def _on_sensor_data(self, raw_data: Dict[str, Any]):
        """
        传感器数据回调函数
        
        数据流:
        1. 接收原始数据
        2. 数据处理
        3. 异常检测
        4. 本地存储
        5. 加入云端上传队列
        """
        self.stats["messages_received"] += 1
        
        try:
            # 1. 数据处理
            processed_data = self.data_processor.process(raw_data)
            
            if processed_data is None:
                self.logger.warning("Data processing failed, skipping")
                return
            
            # 2. 异常检测
            node_id = processed_data.get("node_id", "unknown")
            sensors = processed_data.get("sensors", {})
            
            for sensor_type, sensor_info in sensors.items():
                value = sensor_info.get("value")
                if value is not None:
                    is_anomaly, reasons = self.anomaly_detector.detect(
                        node_id, sensor_type, value
                    )
                    
                    sensor_info["edge_anomaly"] = is_anomaly
                    sensor_info["anomaly_reasons"] = reasons
                    
                    if is_anomaly:
                        self.stats["anomalies_detected"] += 1
                        self.logger.warning(
                            f"Anomaly detected: {node_id}/{sensor_type} = {value}, "
                            f"reasons: {reasons}"
                        )
            
            # 3. 本地存储
            self.local_storage.save_sensor_data(processed_data)
            
            # 4. 加入云端上传队列
            if self.cloud_uploader:
                self.cloud_uploader.add_to_queue(processed_data)
            
            self.stats["messages_processed"] += 1
            self.logger.debug(f"Processed data from {node_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling sensor data: {e}")
    
    def start(self):
        """启动网关"""
        self.logger.info("Starting IoT Gateway...")
        self.running = True
        self.stats["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 设置MQTT回调
        self.mqtt_handler.set_data_callback(self._on_sensor_data)
        
        # 连接MQTT
        if not self.mqtt_handler.connect():
            self.logger.error("Failed to connect to MQTT broker")
            return False
        
        # 订阅主题
        mqtt_topics = self.config.get("mqtt", {}).get("topics", {})
        self.mqtt_handler.subscribe(mqtt_topics.get("sensor_data", "farm/sensors"))
        
        # 启动云端上传器
        if self.cloud_uploader:
            self.cloud_uploader.start()
        
        self.logger.info("IoT Gateway started successfully")
        return True
    
    def stop(self):
        """停止网关"""
        self.logger.info("Stopping IoT Gateway...")
        self.running = False
        
        # 断开MQTT
        self.mqtt_handler.disconnect()
        
        # 停止云端上传器
        if self.cloud_uploader:
            self.cloud_uploader.stop()
        
        self.logger.info("IoT Gateway stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """获取网关状态"""
        return {
            "running": self.running,
            "stats": self.stats,
            "mqtt": self.mqtt_handler.get_stats(),
            "storage": self.local_storage.get_statistics(),
            "cloud": self.cloud_uploader.get_stats() if self.cloud_uploader else None,
            "sensor_health": self.anomaly_detector.get_health_status()
        }
    
    def run_forever(self):
        """运行直到收到停止信号"""
        if not self.start():
            return
        
        # 设置信号处理
        def signal_handler(signum, frame):
            self.logger.info("Received stop signal")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 主循环
        while self.running:
            try:
                time.sleep(1)
                
                # 定期打印状态
                if int(time.time()) % 60 == 0:
                    status = self.get_status()
                    self.logger.info(
                        f"Status: received={status['stats']['messages_received']}, "
                        f"processed={status['stats']['messages_processed']}, "
                        f"anomalies={status['stats']['anomalies_detected']}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")


def main():
    """主入口函数"""
    print("=" * 60)
    print("  TrustFusion-GNN IoT Gateway")
    print("  Version: 1.0.0")
    print("=" * 60)
    
    # 创建并运行网关
    gateway = Gateway()
    gateway.run_forever()


if __name__ == "__main__":
    main()
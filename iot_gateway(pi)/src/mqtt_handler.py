# src/mqtt_handler.py
"""
MQTT处理模块
负责与ESP32传感器节点的MQTT通信
"""

import json
import logging
from typing import Callable, Dict, Any, Optional
from datetime import datetime
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTHandler:
    """
    MQTT客户端处理器
    
    功能:
    1. 连接MQTT Broker
    2. 订阅传感器数据主题
    3. 解析接收到的JSON数据
    4. 调用回调函数处理数据
    """
    
    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        username: str = "",
        password: str = ""
    ):
        """
        初始化MQTT处理器
        
        参数:
            broker: MQTT服务器地址
            port: MQTT端口
            username: 用户名(可选)
            password: 密码(可选)
        """
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        
        # 创建MQTT客户端
        self.client = mqtt.Client(client_id="raspberry_gateway")
        
        # 设置回调函数
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # 设置认证(如果有)
        if username and password:
            self.client.username_pw_set(username, password)
        
        # 数据回调函数(外部设置)
        self.data_callback: Optional[Callable] = None
        
        # 订阅的主题列表
        self.topics: list = []
        
        # 连接状态
        self.connected = False
        
        # 统计信息
        self.stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "errors": 0,
            "last_message_time": None
        }
    
    def connect(self) -> bool:
        """连接到MQTT Broker"""
        try:
            logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}")
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()  # 启动后台线程处理网络
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """断开MQTT连接"""
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def subscribe(self, topic: str):
        """订阅主题"""
        self.topics.append(topic)
        if self.connected:
            self.client.subscribe(topic)
            logger.info(f"Subscribed to topic: {topic}")
    
    def publish(self, topic: str, message: Dict[str, Any]):
        """发布消息"""
        try:
            payload = json.dumps(message)
            self.client.publish(topic, payload)
            logger.debug(f"Published to {topic}: {payload[:100]}...")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
    
    def set_data_callback(self, callback: Callable):
        """设置数据处理回调函数"""
        self.data_callback = callback
    
    def _on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker successfully")
            
            # 重新订阅所有主题
            for topic in self.topics:
                client.subscribe(topic)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect, return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.connected = False
        logger.warning(f"Disconnected from MQTT broker, rc: {rc}")
        
        # 尝试重连
        if rc != 0:
            logger.info("Attempting to reconnect...")
    
    def _on_message(self, client, userdata, msg):
        """消息接收回调"""
        try:
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = datetime.now()
            
            # 解析JSON
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            
            logger.debug(f"Received message on {topic}")
            
            # 添加接收时间戳
            data["gateway_timestamp"] = datetime.now().isoformat()
            data["topic"] = topic
            
            # 调用处理回调
            if self.data_callback:
                self.data_callback(data)
                self.stats["messages_processed"] += 1
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            self.stats["errors"] += 1
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
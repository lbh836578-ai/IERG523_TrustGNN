# src/cloud_uploader.py
"""
云端上传模块
负责将数据上传到云端服务器
"""

import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import time
from threading import Thread, Event

logger = logging.getLogger(__name__)


class CloudUploader:
    """
    云端数据上传器
    
    功能:
    1. 批量上传传感器数据
    2. 自动重试失败的上传
    3. 断网时缓存数据
    4. 后台异步上传
    """
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        upload_interval: int = 30,
        batch_size: int = 50
    ):
        """
        初始化上传器
        
        参数:
            api_url: 云端API地址
            api_key: API密钥
            upload_interval: 上传间隔(秒)
            batch_size: 批量上传大小
        """
        self.api_url = api_url
        self.api_key = api_key
        self.upload_interval = upload_interval
        self.batch_size = batch_size
        
        # 待上传数据队列
        self.upload_queue: List[Dict[str, Any]] = []
        
        # 统计信息
        self.stats = {
            "total_uploaded": 0,
            "total_failed": 0,
            "last_upload_time": None,
            "last_error": None
        }
        
        # 后台上传线程控制
        self._stop_event = Event()
        self._upload_thread: Optional[Thread] = None
    
    def start(self):
        """启动后台上传线程"""
        self._stop_event.clear()
        self._upload_thread = Thread(target=self._upload_loop, daemon=True)
        self._upload_thread.start()
        logger.info("Cloud uploader started")
    
    def stop(self):
        """停止后台上传线程"""
        self._stop_event.set()
        if self._upload_thread:
            self._upload_thread.join(timeout=5)
        logger.info("Cloud uploader stopped")
    
    def add_to_queue(self, data: Dict[str, Any]):
        """添加数据到上传队列"""
        self.upload_queue.append(data)
        logger.debug(f"Data added to upload queue, queue size: {len(self.upload_queue)}")
    
    def _upload_loop(self):
        """后台上传循环"""
        while not self._stop_event.is_set():
            try:
                # 等待指定间隔
                if self._stop_event.wait(self.upload_interval):
                    break
                
                # 检查是否有数据需要上传
                if self.upload_queue:
                    self._do_upload()
                    
            except Exception as e:
                logger.error(f"Error in upload loop: {e}")
    
    def _do_upload(self):
        """执行上传操作"""
        if not self.upload_queue:
            return
        
        # 取出一批数据
        batch = self.upload_queue[:self.batch_size]
        
        try:
            # 构建请求
            payload = {
                "gateway_id": "raspberry_pi_gateway",
                "timestamp": datetime.now().isoformat(),
                "batch_size": len(batch),
                "data": batch
            }
            
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
            
            # 发送请求
            response = requests.post(
                f"{self.api_url}/sensor-data/batch",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                # 上传成功，从队列中移除
                self.upload_queue = self.upload_queue[len(batch):]
                self.stats["total_uploaded"] += len(batch)
                self.stats["last_upload_time"] = datetime.now().isoformat()
                logger.info(f"Successfully uploaded {len(batch)} records")
                
                # 处理响应（可能包含服务器的分析结果）
                result = response.json()
                self._handle_response(result)
                
            else:
                # 上传失败
                self.stats["total_failed"] += len(batch)
                self.stats["last_error"] = f"HTTP {response.status_code}"
                logger.error(f"Upload failed: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot connect to cloud server, data queued for later")
            self.stats["last_error"] = "Connection error"
            
        except requests.exceptions.Timeout:
            logger.warning("Upload timed out, will retry later")
            self.stats["last_error"] = "Timeout"
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            self.stats["last_error"] = str(e)
    
    def _handle_response(self, result: Dict[str, Any]):
        """处理云端返回的结果"""
        # 检查是否有异常警报
        if "alerts" in result:
            for alert in result["alerts"]:
                logger.warning(f"Cloud alert: {alert}")
        
        # 检查是否有控制命令
        if "commands" in result:
            for cmd in result["commands"]:
                logger.info(f"Cloud command received: {cmd}")
                # TODO: 执行控制命令
    
    def upload_now(self) -> bool:
        """立即上传所有队列中的数据"""
        if not self.upload_queue:
            return True
        
        original_size = len(self.upload_queue)
        
        while self.upload_queue:
            self._do_upload()
            
            # 如果队列没有减少，说明上传失败
            if len(self.upload_queue) >= original_size:
                return False
            
            original_size = len(self.upload_queue)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取上传统计信息"""
        return {
            **self.stats,
            "queue_size": len(self.upload_queue)
        }
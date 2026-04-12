# src/local_storage.py
"""
本地存储模块
使用SQLite数据库存储传感器数据
"""

import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalStorage:
    """
    本地SQLite存储
    
    功能:
    1. 存储所有接收到的传感器数据
    2. 支持按时间和节点查询
    3. 自动清理过期数据
    4. 断网时缓存未上传的数据
    """
    
    def __init__(
        self, 
        db_path: str = "data/sensor_data.db",
        max_records: int = 100000
    ):
        """
        初始化存储
        
        参数:
            db_path: 数据库文件路径
            max_records: 最大记录数
        """
        self.db_path = db_path
        self.max_records = max_records
        
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建传感器数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                value REAL,
                quality REAL,
                is_anomaly INTEGER,
                raw_data TEXT,
                uploaded INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_node_timestamp 
            ON sensor_data(node_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_uploaded 
            ON sensor_data(uploaded)
        """)
        
        # 创建系统日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT,
                message TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_sensor_data(self, data: Dict[str, Any]) -> bool:
        """
        保存传感器数据
        
        参数:
            data: 处理后的传感器数据
            
        返回:
            是否保存成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            node_id = data.get("node_id", "unknown")
            timestamp = data.get("timestamp", datetime.now().isoformat())
            
            # 保存每个传感器的数据
            sensors = data.get("sensors", {})
            for sensor_type, sensor_info in sensors.items():
                cursor.execute("""
                    INSERT INTO sensor_data 
                    (node_id, timestamp, sensor_type, value, quality, 
                     is_anomaly, raw_data, uploaded)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    node_id,
                    timestamp,
                    sensor_type,
                    sensor_info.get("value"),
                    sensor_info.get("quality", 1.0),
                    1 if sensor_info.get("is_anomaly", False) else 0,
                    json.dumps(sensor_info)
                ))
            
            conn.commit()
            conn.close()
            
            # 检查是否需要清理旧数据
            self._cleanup_if_needed()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sensor data: {e}")
            return False
    
    def get_unuploaded_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取未上传的数据
        
        参数:
            limit: 最大返回记录数
            
        返回:
            未上传的数据列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, node_id, timestamp, sensor_type, 
                       value, quality, is_anomaly
                FROM sensor_data
                WHERE uploaded = 0
                ORDER BY timestamp ASC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # 按时间戳分组数据
            data_dict = {}
            for row in rows:
                id_, node_id, timestamp, sensor_type, value, quality, is_anomaly = row
                
                key = f"{node_id}_{timestamp}"
                if key not in data_dict:
                    data_dict[key] = {
                        "ids": [],
                        "node_id": node_id,
                        "timestamp": timestamp,
                        "sensors": {}
                    }
                
                data_dict[key]["ids"].append(id_)
                data_dict[key]["sensors"][sensor_type] = {
                    "value": value,
                    "quality": quality,
                    "is_anomaly": bool(is_anomaly)
                }
            
            return list(data_dict.values())
            
        except Exception as e:
            logger.error(f"Failed to get unuploaded data: {e}")
            return []
    
    def mark_as_uploaded(self, record_ids: List[int]):
        """标记数据为已上传"""
        if not record_ids:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ",".join(["?" for _ in record_ids])
            cursor.execute(f"""
                UPDATE sensor_data
                SET uploaded = 1
                WHERE id IN ({placeholders})
            """, record_ids)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Marked {len(record_ids)} records as uploaded")
            
        except Exception as e:
            logger.error(f"Failed to mark data as uploaded: {e}")
    
    def get_recent_data(
        self, 
        node_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        获取最近的数据
        
        参数:
            node_id: 节点ID(可选)
            sensor_type: 传感器类型(可选)
            hours: 时间范围(小时)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            query = """
                SELECT node_id, timestamp, sensor_type, value, quality
                FROM sensor_data
                WHERE timestamp > ?
            """
            params = [since]
            
            if node_id:
                query += " AND node_id = ?"
                params.append(node_id)
            
            if sensor_type:
                query += " AND sensor_type = ?"
                params.append(sensor_type)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "node_id": row[0],
                    "timestamp": row[1],
                    "sensor_type": row[2],
                    "value": row[3],
                    "quality": row[4]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Failed to get recent data: {e}")
            return []
    
    def _cleanup_if_needed(self):
        """清理过期数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取当前记录数
            cursor.execute("SELECT COUNT(*) FROM sensor_data")
            count = cursor.fetchone()[0]
            
            if count > self.max_records:
                # 删除最老的已上传数据
                delete_count = count - self.max_records + 1000
                cursor.execute("""
                    DELETE FROM sensor_data
                    WHERE id IN (
                        SELECT id FROM sensor_data
                        WHERE uploaded = 1
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """, (delete_count,))
                
                conn.commit()
                logger.info(f"Cleaned up {delete_count} old records")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to cleanup database: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 总记录数
            cursor.execute("SELECT COUNT(*) FROM sensor_data")
            total = cursor.fetchone()[0]
            
            # 未上传数
            cursor.execute("SELECT COUNT(*) FROM sensor_data WHERE uploaded = 0")
            unuploaded = cursor.fetchone()[0]
            
            # 各节点记录数
            cursor.execute("""
                SELECT node_id, COUNT(*) 
                FROM sensor_data 
                GROUP BY node_id
            """)
            by_node = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_records": total,
                "unuploaded": unuploaded,
                "by_node": by_node,
                "database_path": self.db_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
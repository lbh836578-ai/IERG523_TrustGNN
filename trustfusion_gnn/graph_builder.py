"""
图构建模块
构建传感器空间关系图 A ∈ ℝ^{N×N}
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import SystemConfig, SensorMeta, SensorType


class GraphBuilder:
    """
    图构建器
    基于物理距离和传感器类型构建邻接矩阵
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.sensor_ids = list(config.sensors.keys())
        self.num_sensors = len(self.sensor_ids)
        
        # 预计算各种图
        self._distance_adj = None
        self._type_adj = None
        self._correlation_adj = None
        self._combined_adj = None
        
    def get_distance_adjacency(self) -> torch.Tensor:
        """
        基于物理距离构建邻接矩阵
        使用高斯核: A_ij = exp(-d_ij^2 / (2*sigma^2))
        """
        if self._distance_adj is not None:
            return self._distance_adj
            
        N = self.num_sensors
        adj = torch.zeros(N, N)
        
        # 获取位置
        positions = []
        for sid in self.sensor_ids:
            sensor = self.config.sensors[sid]
            positions.append(sensor.position)
        positions = np.array(positions)  # (N, 3)
        
        # 计算距离矩阵
        for i in range(N):
            for j in range(N):
                dist = np.linalg.norm(positions[i] - positions[j])
                # 高斯核
                sigma = 3.0  # 可调参数
                adj[i, j] = np.exp(-dist**2 / (2 * sigma**2))
        
        # 归一化
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        
        self._distance_adj = adj
        return adj
    
    def get_type_adjacency(self) -> torch.Tensor:
        """
        基于传感器类型构建邻接矩阵
        同类型传感器有更强连接
        """
        if self._type_adj is not None:
            return self._type_adj
            
        N = self.num_sensors
        adj = torch.zeros(N, N)
        
        # 获取类型
        types = []
        for sid in self.sensor_ids:
            sensor = self.config.sensors[sid]
            types.append(sensor.sensor_type)
        
        # 同类型连接强度高
        for i in range(N):
            for j in range(N):
                if types[i] == types[j]:
                    adj[i, j] = 1.0
                else:
                    # 不同类型也有弱连接（物理相关性）
                    adj[i, j] = 0.2
        
        # 归一化
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        
        self._type_adj = adj
        return adj
    
    def get_esp32_adjacency(self) -> torch.Tensor:
        """
        基于 ESP32 分组构建邻接矩阵
        同一 ESP32 上的传感器有更强连接（可能共享故障模式）
        """
        N = self.num_sensors
        adj = torch.zeros(N, N)
        
        # 获取 ESP32 分组
        esp32_ids = []
        for sid in self.sensor_ids:
            sensor = self.config.sensors[sid]
            esp32_ids.append(sensor.esp32_id)
        
        for i in range(N):
            for j in range(N):
                if esp32_ids[i] == esp32_ids[j]:
                    adj[i, j] = 1.0
                else:
                    adj[i, j] = 0.3
        
        # 归一化
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
        
        return adj
    
    def get_combined_adjacency(
        self,
        distance_weight: float = 0.4,
        type_weight: float = 0.4,
        esp32_weight: float = 0.2
    ) -> torch.Tensor:
        """
        组合多种邻接矩阵
        """
        if self._combined_adj is not None:
            return self._combined_adj
            
        adj_dist = self.get_distance_adjacency()
        adj_type = self.get_type_adjacency()
        adj_esp32 = self.get_esp32_adjacency()
        
        combined = (
            distance_weight * adj_dist +
            type_weight * adj_type +
            esp32_weight * adj_esp32
        )
        
        # 归一化
        combined = combined / (combined.sum(dim=1, keepdim=True) + 1e-8)
        
        # 添加自环
        combined = 0.5 * combined + 0.5 * torch.eye(self.num_sensors)
        
        self._combined_adj = combined
        return combined
    
    def get_k_nearest_neighbors(self, k: int = 3) -> torch.Tensor:
        """
        基于距离的 K 近邻图
        """
        adj = self.get_distance_adjacency()
        
        # 保留每个节点的 top-k 连接
        _, indices = adj.topk(k + 1, dim=1)  # +1 是因为包含自身
        
        mask = torch.zeros_like(adj)
        for i in range(self.num_sensors):
            mask[i, indices[i]] = 1.0
        
        # 对称化
        mask = (mask + mask.T) / 2
        mask = (mask > 0).float()
        
        # 应用掩码
        adj_knn = adj * mask
        
        # 归一化
        adj_knn = adj_knn / (adj_knn.sum(dim=1, keepdim=True) + 1e-8)
        
        return adj_knn
    
    def visualize_graph(self):
        """可视化图结构（可选）"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            adj = self.get_combined_adjacency().numpy()
            
            G = nx.from_numpy_array(adj)
            
            # 设置节点标签
            labels = {i: self.sensor_ids[i] for i in range(self.num_sensors)}
            
            # 设置节点颜色（按类型）
            colors = []
            color_map = {
                SensorType.TEMPERATURE: 'red',
                SensorType.HUMIDITY: 'blue',
                SensorType.SOIL_MOISTURE: 'brown',
                SensorType.LIGHT: 'yellow'
            }
            for sid in self.sensor_ids:
                sensor = self.config.sensors[sid]
                colors.append(color_map.get(sensor.sensor_type, 'gray'))
            
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(
                G, pos,
                labels=labels,
                node_color=colors,
                node_size=1000,
                font_size=8,
                width=[adj[i, j] * 3 for i, j in G.edges()]
            )
            plt.title("Sensor Graph Structure")
            plt.savefig("sensor_graph.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("图结构已保存到 sensor_graph.png")
            
        except ImportError:
            print("需要安装 matplotlib 和 networkx 来可视化图")
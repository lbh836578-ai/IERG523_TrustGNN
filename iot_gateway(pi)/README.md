# 在树莓派上执行

# 1. 进入项目目录
cd ~/iot_gateway(pi)

# 2. 安装依赖
pip3 install -r requirements.txt

# 3. 创建必要的目录
mkdir -p config logs data

# 4. 修改配置文件
nano config/config.yaml
# 根据实际情况修改MQTT和云端配置

# 5. 运行网关
python3 main.py

# 6. 设置开机自启（可选）
# 创建服务文件
sudo nano /etc/systemd/system/iot-gateway.service

# 添加以下内容:
# [Unit]
# Description=IoT Gateway Service
# After=network.target
# 
# [Service]
# Type=simple
# User=pi
# WorkingDirectory=/home/pi/iot_gateway
# ExecStart=/usr/bin/python3 /home/pi/iot_gateway/main.py
# Restart=always
# RestartSec=10
# 
# [Install]
# WantedBy=multi-user.target

# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable iot-gateway(pi)
sudo systemctl start iot-gateway(pi)

# 查看服务状态
sudo systemctl status iot-gateway

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TrustFusion-GNN 完整数据流                             │
└─────────────────────────────────────────────────────────────────────────────────┘

    ESP32节点1        ESP32节点2        ESP32节点3        ESP32节点4
         │                 │                 │                 │
         │ DHT22          │ DHT22          │ DHT22          │ DHT22
         │ 土壤           │ 土壤           │ 土壤           │ 土壤
         │ BH1750         │ BH1750         │ BH1750         │ BH1750
         │                 │                 │                 │
         └────────┬────────┴────────┬────────┴────────┬────────┘
                  │                 │                 │
                  │    WiFi/MQTT    │                 │
                  │                 │                 │
                  ▼                 ▼                 ▼
         ┌─────────────────────────────────────────────────────┐
         │                  树莓派网关                          │
         │                                                     │
         │   ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
         │   │MQTT Handler │─→│Data Process │─→│Anomaly Det │  │
         │   └─────────────┘  └─────────────┘  └────────────┘  │
         │          │                │                │        │
         │          │                ▼                │        │
         │          │         ┌────────────┐         │        │
         │          │         │Local Store │         │        │
         │          │         │ (SQLite)   │         │        │
         │          │         └────────────┘         │        │
         │          │                │                │        │
         │          └────────────────┼────────────────┘        │
         │                           │                         │
         │                           ▼                         │
         │                  ┌─────────────────┐                │
         │                  │ Cloud Uploader  │                │
         │                  └─────────────────┘                │
         │                           │                         │
         └───────────────────────────┼─────────────────────────┘
                                     │
                                     │ HTTP/REST API
                                     │
                                     ▼
         ┌─────────────────────────────────────────────────────┐
         │                     云端服务器                       │
         │                                                     │
         │   ┌─────────────────────────────────────────────┐   │
         │   │              TrustFusion-GNN                │   │
         │   │                                             │   │
         │   │   Stage1         Stage2         Stage3      │   │
         │   │   LSTM    ──→   GNN     ──→   Attention    │   │
         │   │   时序特征       空间特征       可信融合     │   │
         │   │                                             │   │
         │   └─────────────────────────────────────────────┘   │
         │                         │                           │
         │                         ▼                           │
         │   ┌─────────────────────────────────────────────┐   │
         │   │              输出结果                        │   │
         │   │  - 融合后的传感器值                          │   │
         │   │  - 各传感器可信度评分                        │   │
         │   │  - 不确定性估计                              │   │
         │   │  - 异常警报                                  │   │
         │   └─────────────────────────────────────────────┘   │
         │                         │                           │
         └─────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │   可视化界面    │
                          │   决策支持      │
                          │   报警系统      │
                          └─────────────────┘
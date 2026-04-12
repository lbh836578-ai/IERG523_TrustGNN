"""
主程序入口
TrustFusion-GNN 完整演示
"""
import torch
import numpy as np
from datetime import datetime
from typing import Dict
import os

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


def print_banner():
    print("=" * 70)
    print("  TrustFusion-GNN: 可信数据融合网络")
    print("  GNN-Enabled Trustworthy Sensor Fusion for Smart Agriculture")
    print("=" * 70)
    print(f"\n运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def demo_model_architecture():
    """演示模型架构"""
    from config import get_agricultural_config
    from models.trustfusion_gnn import TrustFusionGNN
    
    print("\n" + "="*60)
    print("1. 模型架构演示")
    print("="*60)
    
    config = get_agricultural_config()
    
    # 创建模型
    model = TrustFusionGNN(
        num_sensors=config.num_sensors,
        input_dim=config.input_features,
        output_dim=config.output_features,
        hidden_dim=config.gnn_hidden_dim,
        temporal_layers=config.temporal_layers,
        gnn_layers=config.gnn_layers,
        num_heads=config.num_attention_heads,
        dropout=config.dropout
    )
    
    # 打印模型信息
    summary = model.get_model_summary()
    print(f"\n模型参数:")
    print(f"  总参数量: {summary['total_parameters']:,}")
    print(f"  可训练参数: {summary['trainable_parameters']:,}")
    print(f"  模型大小: {summary['model_size_mb']:.2f} MB")
    
    # 测试前向传播
    print("\n测试前向传播:")
    B, N, T, F = 2, config.num_sensors, config.window_size, config.input_features
    X = torch.randn(B, N, T, F)
    
    # 构建邻接矩阵
    from graph_builder import GraphBuilder
    graph_builder = GraphBuilder(config)
    A = graph_builder.get_combined_adjacency()
    
    print(f"  输入 X shape: {X.shape} (batch={B}, N={N}, T={T}, F={F})")
    print(f"  邻接矩阵 A shape: {A.shape}")
    
    with torch.no_grad():
        output = model(X, A)
    
    print(f"\n  输出:")
    print(f"    Ŷ (融合结果): {output.Y_hat.shape} → (batch, T, output_features)")
    print(f"    τ (可信度): {output.tau.shape} → (batch, N)")
    print(f"    τ_full (时变可信度): {output.tau_full.shape} → (batch, N, T)")
    print(f"    σ (不确定性): {output.sigma.shape} → (batch, T, output_features)")
    print(f"    anomaly_flags: {output.anomaly_flags.shape} → (batch, N)")
    print(f"    anomaly_scores: {output.anomaly_scores.shape} → (batch, N)")
    print(f"    system_confidence: {output.system_confidence.shape} → (batch,)")
    
    return model, config


def demo_data_simulation():
    """演示数据模拟"""
    from config import get_agricultural_config
    from data_simulator import AgriculturalDataSimulator
    
    print("\n" + "="*60)
    print("2. 数据模拟演示")
    print("="*60)
    
    config = get_agricultural_config()
    simulator = AgriculturalDataSimulator(config, seed=42)
    
    # 生成数据
    print("\n生成模拟数据...")
    data_list, gt_list, fault_list = simulator.generate_dataset(
        num_samples=5,
        inject_faults=True,
        fault_ratio=0.4
    )
    
    print(f"  生成样本数: {len(data_list)}")
    print(f"  每个样本形状: {data_list[0].X.shape} (N, T, F)")
    print(f"  融合目标形状: {gt_list[0].fusion_target.shape} (T, output_F)")
    
    # 展示故障信息
    print("\n故障注入示例:")
    for i, faults in enumerate(fault_list[:3]):
        if faults:
            print(f"\n  样本 {i+1}:")
            for f in faults:
                print(f"    - {f.sensor_id}: {f.fault_type.name} "
                      f"(时间: {f.start_time}-{f.end_time})")
        else:
            print(f"\n  样本 {i+1}: 无故障")
    
    return simulator, config


def demo_graph_structure():
    """演示图结构"""
    from config import get_agricultural_config
    from graph_builder import GraphBuilder
    
    print("\n" + "="*60)
    print("3. 图结构演示")
    print("="*60)
    
    config = get_agricultural_config()
    graph_builder = GraphBuilder(config)
    
    # 获取各种邻接矩阵
    adj_dist = graph_builder.get_distance_adjacency()
    adj_type = graph_builder.get_type_adjacency()
    adj_combined = graph_builder.get_combined_adjacency()
    
    print("\n传感器列表:")
    for i, sid in enumerate(graph_builder.sensor_ids):
        sensor = config.sensors[sid]
        print(f"  {i}: {sid} ({sensor.sensor_type.value}) @ ESP32-{sensor.esp32_id}")
    
    print(f"\n邻接矩阵形状: {adj_combined.shape}")
    print(f"\n组合邻接矩阵 (前4×4):\n{adj_combined[:4, :4].numpy().round(3)}")
    
    return graph_builder


def run_training():
    """运行训练"""
    from config import get_agricultural_config
    from data_simulator import AgriculturalDataSimulator
    from models.trustfusion_gnn import TrustFusionGNN
    from trainer import Trainer
    
    print("\n" + "="*60)
    print("4. 模型训练")
    print("="*60)
    
    config = get_agricultural_config()
    
    # 生成数据
    print("\n生成训练数据...")
    simulator = AgriculturalDataSimulator(config, seed=42)
    
    train_data, train_gt, _ = simulator.generate_dataset(
        num_samples=200, 
        inject_faults=True,
        fault_ratio=0.4
    )
    val_data, val_gt, _ = simulator.generate_dataset(
        num_samples=50,
        inject_faults=True,
        fault_ratio=0.4
    )
    
    print(f"  训练样本: {len(train_data)}")
    print(f"  验证样本: {len(val_data)}")
    
    # 创建数据加载器
    train_loader = simulator.create_dataloader(
        train_data, train_gt, batch_size=16, shuffle=True
    )
    val_loader = simulator.create_dataloader(
        val_data, val_gt, batch_size=16, shuffle=False
    )
    
    # 创建模型
    model = TrustFusionGNN(
        num_sensors=config.num_sensors,
        input_dim=config.input_features,
        output_dim=config.output_features,
        hidden_dim=32,  # 减小以加速演示
        temporal_layers=1,
        gnn_layers=1,
        num_heads=2,
        dropout=0.1
    )
    
    # 训练
    trainer = Trainer(model, config)
    history = trainer.train(
        train_loader, 
        val_loader,
        num_epochs=15,
        patience=5
    )
    
    # 最终评估
    print("\n" + "-"*40)
    print("最终评估")
    print("-"*40)
    
    val_losses, metrics = trainer.evaluate(val_loader)
    
    print(f"\n融合精度:")
    print(f"  MAE: {metrics.mae:.4f}")
    print(f"  RMSE: {metrics.rmse:.4f}")
    print(f"  MAPE: {metrics.mape:.2f}%")
    
    print(f"\n异常检测:")
    print(f"  AUC: {metrics.anomaly_auc:.4f}")
    print(f"  Precision: {metrics.anomaly_precision:.4f}")
    print(f"  Recall: {metrics.anomaly_recall:.4f}")
    print(f"  F1: {metrics.anomaly_f1:.4f}")
    
    print(f"\n系统指标:")
    print(f"  可信度 MAE: {metrics.credibility_mae:.4f}")
    print(f"  平均系统置信度: {metrics.system_confidence_mean:.4f}")
    
    return model, trainer, config


def demo_inference(model, config):
    """演示推理"""
    from data_simulator import AgriculturalDataSimulator
    from inference import InferenceEngine
    from graph_builder import GraphBuilder
    
    print("\n" + "="*60)
    print("5. 推理演示")
    print("="*60)
    
    # 创建推理引擎
    engine = InferenceEngine(model, config)
    
    # 生成测试数据
    simulator = AgriculturalDataSimulator(config, seed=123)
    test_data, test_gt, test_faults = simulator.generate_dataset(
        num_samples=3,
        inject_faults=True,
        fault_ratio=0.5
    )
    
    sensor_ids = list(config.sensors.keys())
    output_names = ['温度', '湿度', '土壤湿度', '光照']
    output_units = ['°C', '%RH', '%', 'lux']
    
    for i, (data, gt, faults) in enumerate(zip(test_data, test_gt, test_faults)):
        print(f"\n{'='*50}")
        print(f"测试样本 {i+1}")
        print(f"{'='*50}")
        
        # 推理
        result = engine.process_window(data.X.numpy())
        
        # 显示融合结果
        print("\n📊 融合结果 Ŷ:")
        for j, (name, unit) in enumerate(zip(output_names, output_units)):
            key = ['temperature', 'humidity', 'soil_moisture', 'light'][j]
            val = result.fused_values[key]
            unc = result.uncertainties[key]
            print(f"  {name}: {val:.2f} ± {unc:.2f} {unit}")
        
        # 显示传感器可信度
        print("\n🔍 传感器可信度 τ:")
        for sid in sensor_ids:
            tau = result.sensor_credibility[sid]
            is_anomaly = result.anomaly_flags[sid]
            status = "⚠️ 异常" if is_anomaly else "✓ 正常"
            bar = "█" * int(tau * 10) + "░" * (10 - int(tau * 10))
            print(f"  {sid:12s}: [{bar}] {tau:.3f} {status}")
        
        # 系统置信度
        print(f"\n🎯 系统整体置信度: {result.system_confidence:.3f}")
        
        # 报警
        if result.alerts:
            print("\n⚠️ 报警信息:")
            for alert in result.alerts:
                print(f"  - {alert}")
        
        if result.recommendations:
            print("\n💡 建议:")
            for rec in result.recommendations:
                print(f"  - {rec}")
        
        # 实际故障（如果有）
        if faults:
            print(f"\n📝 实际注入的故障:")
            for f in faults:
                print(f"  - {f.sensor_id}: {f.fault_type.name}")


def run_robustness_test(model, config):
    """运行鲁棒性测试"""
    from data_simulator import AgriculturalDataSimulator
    from inference import RobustnessEvaluator
    
    print("\n" + "="*60)
    print("6. 鲁棒性测试")
    print("="*60)
    
    # 生成干净测试数据
    simulator = AgriculturalDataSimulator(config, seed=999)
    clean_samples = simulator.generate_clean_data(num_samples=50)
    
    # 计算融合目标（与训练标签口径一致）
    clean_X = torch.stack([s.X for s in clean_samples])  # (50, N, T, F)
    fusion_target_list = []
    for sample in clean_samples:
        fault_mask = torch.zeros(sample.X.shape[0], sample.X.shape[1])
        gt = simulator.compute_ground_truth(sample, fault_mask)
        fusion_target_list.append(gt.fusion_target)
    fusion_target = torch.stack(fusion_target_list, dim=0)  # (50, T, 4)
    
    # 评估器
    evaluator = RobustnessEvaluator(model, config)
    
    print("\n不同异常比例下的性能:")
    print("-" * 40)
    
    anomaly_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = evaluator.evaluate_robustness(
        clean_X, fusion_target, anomaly_ratios
    )
    
    print("\n📈 鲁棒性总结:")
    print("-" * 40)
    baseline_mae = results[0.0]['mae']
    for ratio, res in results.items():
        degradation = (res['mae'] - baseline_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        print(f"  异常比例 {ratio*100:4.0f}%: MAE={res['mae']:.4f} (退化 {degradation:+.1f}%)")


def main():
    print_banner()
    
    print("请选择运行模式:")
    print("  1. 仅模型架构演示")
    print("  2. 数据模拟演示")
    print("  3. 图结构演示")
    print("  4. 快速训练 + 推理")
    print("  5. 完整流程（含鲁棒性测试）")
    
    choice = input("\n输入选择 (1-5，默认5): ").strip() or "5"
    
    if choice == "1":
        demo_model_architecture()
        
    elif choice == "2":
        demo_data_simulation()
        
    elif choice == "3":
        demo_graph_structure()
        
    elif choice == "4":
        model, trainer, config = run_training()
        demo_inference(model, config)
        
    else:
        # 完整流程
        demo_model_architecture()
        demo_data_simulation()
        demo_graph_structure()
        model, trainer, config = run_training()
        demo_inference(model, config)
        run_robustness_test(model, config)
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)


if __name__ == "__main__":
    main()
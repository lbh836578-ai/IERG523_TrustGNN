"""
Main entry point
Full TrustFusion-GNN demo
"""
import torch
import numpy as np
from datetime import datetime
from typing import Dict
import os

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


def print_banner():
    print("=" * 70)
    print("  TrustFusion-GNN: Trustworthy Data Fusion Network")
    print("  GNN-Enabled Trustworthy Sensor Fusion for Smart Agriculture")
    print("=" * 70)
    print(f"\nRun time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def demo_model_architecture():
    """Demonstrate model architecture"""
    from config import get_agricultural_config
    from models.trustfusion_gnn import TrustFusionGNN
    
    print("\n" + "="*60)
    print("1. Model Architecture Demo")
    print("="*60)
    
    config = get_agricultural_config()
    
    # Create model
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
    
    # Print model information
    summary = model.get_model_summary()
    print(f"\nModel parameters:")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")
    print(f"  Model size: {summary['model_size_mb']:.2f} MB")
    
    # Test forward pass
    print("\nForward pass test:")
    B, N, T, F = 2, config.num_sensors, config.window_size, config.input_features
    X = torch.randn(B, N, T, F)
    
    # Build adjacency matrix
    from graph_builder import GraphBuilder
    graph_builder = GraphBuilder(config)
    A = graph_builder.get_combined_adjacency()
    
    print(f"  Input X shape: {X.shape} (batch={B}, N={N}, T={T}, F={F})")
    print(f"  Adjacency matrix A shape: {A.shape}")
    
    with torch.no_grad():
        output = model(X, A)
    
    print(f"\n  Outputs:")
    print(f"    Ŷ (fused output): {output.Y_hat.shape} -> (batch, T, output_features)")
    print(f"    τ (trust score): {output.tau.shape} -> (batch, N)")
    print(f"    τ_full (time-varying trust): {output.tau_full.shape} -> (batch, N, T)")
    print(f"    σ (uncertainty): {output.sigma.shape} -> (batch, T, output_features)")
    print(f"    anomaly_flags: {output.anomaly_flags.shape} → (batch, N)")
    print(f"    anomaly_scores: {output.anomaly_scores.shape} → (batch, N)")
    print(f"    system_confidence: {output.system_confidence.shape} → (batch,)")
    
    return model, config


def demo_data_simulation():
    """Demonstrate data simulation"""
    from config import get_agricultural_config
    from data_simulator import AgriculturalDataSimulator
    
    print("\n" + "="*60)
    print("2. Data Simulation Demo")
    print("="*60)
    
    config = get_agricultural_config()
    simulator = AgriculturalDataSimulator(config, seed=42)
    
    # Generate data
    print("\nGenerating synthetic data...")
    data_list, gt_list, fault_list = simulator.generate_dataset(
        num_samples=5,
        inject_faults=True,
        fault_ratio=0.4
    )
    
    print(f"  Number of samples: {len(data_list)}")
    print(f"  Shape per sample: {data_list[0].X.shape} (N, T, F)")
    print(f"  Fusion target shape: {gt_list[0].fusion_target.shape} (T, output_F)")
    
    # Show injected fault information
    print("\nFault injection examples:")
    for i, faults in enumerate(fault_list[:3]):
        if faults:
            print(f"\n  Sample {i+1}:")
            for f in faults:
                print(f"    - {f.sensor_id}: {f.fault_type.name} "
                      f"(time: {f.start_time}-{f.end_time})")
        else:
            print(f"\n  Sample {i+1}: no fault")
    
    return simulator, config


def demo_graph_structure():
    """Demonstrate graph structure"""
    from config import get_agricultural_config
    from graph_builder import GraphBuilder
    
    print("\n" + "="*60)
    print("3. Graph Structure Demo")
    print("="*60)
    
    config = get_agricultural_config()
    graph_builder = GraphBuilder(config)
    
    # Get adjacency variants
    adj_dist = graph_builder.get_distance_adjacency()
    adj_type = graph_builder.get_type_adjacency()
    adj_combined = graph_builder.get_combined_adjacency()
    
    print("\nSensor list:")
    for i, sid in enumerate(graph_builder.sensor_ids):
        sensor = config.sensors[sid]
        print(f"  {i}: {sid} ({sensor.sensor_type.value}) @ ESP32-{sensor.esp32_id}")
    
    print(f"\nAdjacency shape: {adj_combined.shape}")
    print(f"\nCombined adjacency (top-left 4x4):\n{adj_combined[:4, :4].numpy().round(3)}")
    
    return graph_builder


def run_training():
    """Run training"""
    from config import get_agricultural_config
    from data_simulator import AgriculturalDataSimulator
    from models.trustfusion_gnn import TrustFusionGNN
    from trainer import Trainer
    
    print("\n" + "="*60)
    print("4. Model Training")
    print("="*60)
    
    config = get_agricultural_config()
    
    # Generate data
    print("\nGenerating training data...")
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
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    # Create dataloaders
    train_loader = simulator.create_dataloader(
        train_data, train_gt, batch_size=16, shuffle=True
    )
    val_loader = simulator.create_dataloader(
        val_data, val_gt, batch_size=16, shuffle=False
    )
    
    # Create model
    model = TrustFusionGNN(
        num_sensors=config.num_sensors,
        input_dim=config.input_features,
        output_dim=config.output_features,
        hidden_dim=32,  # reduced for faster demo
        temporal_layers=1,
        gnn_layers=1,
        num_heads=2,
        dropout=0.1
    )
    
    # Train
    trainer = Trainer(model, config)
    history = trainer.train(
        train_loader, 
        val_loader,
        num_epochs=15,
        patience=5
    )
    
    # Final evaluation
    print("\n" + "-"*40)
    print("Final Evaluation")
    print("-"*40)
    
    val_losses, metrics = trainer.evaluate(val_loader)
    
    print(f"\nFusion Accuracy:")
    print(f"  MAE: {metrics.mae:.4f}")
    print(f"  RMSE: {metrics.rmse:.4f}")
    print(f"  MAPE: {metrics.mape:.2f}%")
    print("  Per-channel MAE:")
    for channel, value in metrics.per_channel_mae.items():
        print(f"    {channel}: {value:.4f}")
    print("  Per-channel RMSE:")
    for channel, value in metrics.per_channel_rmse.items():
        print(f"    {channel}: {value:.4f}")
    
    print(f"\nAnomaly Detection:")
    print(f"  AUC: {metrics.anomaly_auc:.4f}")
    print(f"  Precision: {metrics.anomaly_precision:.4f}")
    print(f"  Recall: {metrics.anomaly_recall:.4f}")
    print(f"  F1: {metrics.anomaly_f1:.4f}")
    
    print(f"\nSystem Metrics:")
    print(f"  Trust-score MAE: {metrics.credibility_mae:.4f}")
    print(f"  Mean system confidence: {metrics.system_confidence_mean:.4f}")
    
    return model, trainer, config


def demo_inference(model, config):
    """Demonstrate inference"""
    from data_simulator import AgriculturalDataSimulator
    from inference import InferenceEngine
    from graph_builder import GraphBuilder
    
    print("\n" + "="*60)
    print("5. Inference Demo")
    print("="*60)
    
    # Create inference engine
    engine = InferenceEngine(model, config)
    
    # Generate test data
    simulator = AgriculturalDataSimulator(config, seed=123)
    test_data, test_gt, test_faults = simulator.generate_dataset(
        num_samples=3,
        inject_faults=True,
        fault_ratio=0.5
    )
    
    sensor_ids = list(config.sensors.keys())
    output_names = ['Temperature', 'Humidity', 'Soil Moisture', 'Light']
    output_units = ['°C', '%RH', '%', 'lux']
    
    for i, (data, gt, faults) in enumerate(zip(test_data, test_gt, test_faults)):
        print(f"\n{'='*50}")
        print(f"Test Sample {i+1}")
        print(f"{'='*50}")
        
        # Inference
        result = engine.process_window(data.X.numpy())
        
        # Show fusion results
        print("\nFusion Output Ŷ:")
        for j, (name, unit) in enumerate(zip(output_names, output_units)):
            key = ['temperature', 'humidity', 'soil_moisture', 'light'][j]
            val = result.fused_values[key]
            unc = result.uncertainties[key]
            print(f"  {name}: {val:.2f} ± {unc:.2f} {unit}")
        
        # Show sensor trust scores
        print("\nSensor Trust Scores τ:")
        for sid in sensor_ids:
            tau = result.sensor_credibility[sid]
            is_anomaly = result.anomaly_flags[sid]
            status = "[ANOMALY]" if is_anomaly else "[NORMAL]"
            bar = "█" * int(tau * 10) + "░" * (10 - int(tau * 10))
            print(f"  {sid:12s}: [{bar}] {tau:.3f} {status}")
        
        # System confidence
        print(f"\nOverall System Confidence: {result.system_confidence:.3f}")
        
        # Alerts
        if result.alerts:
            print("\nAlerts:")
            for alert in result.alerts:
                print(f"  - {alert}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
        
        # Actual injected faults (if any)
        if faults:
            print(f"\nInjected faults:")
            for f in faults:
                print(f"  - {f.sensor_id}: {f.fault_type.name}")


def run_robustness_test(model, config):
    """Run robustness test"""
    from data_simulator import AgriculturalDataSimulator
    from inference import RobustnessEvaluator
    
    print("\n" + "="*60)
    print("6. Robustness Test")
    print("="*60)
    
    # Generate clean test data
    simulator = AgriculturalDataSimulator(config, seed=999)
    clean_samples = simulator.generate_clean_data(num_samples=50)
    
    # Compute fusion target (consistent with training labels)
    clean_X = torch.stack([s.X for s in clean_samples])  # (50, N, T, F)
    fusion_target_list = []
    for sample in clean_samples:
        fault_mask = torch.zeros(sample.X.shape[0], sample.X.shape[1])
        gt = simulator.compute_ground_truth(sample, fault_mask)
        fusion_target_list.append(gt.fusion_target)
    fusion_target = torch.stack(fusion_target_list, dim=0)  # (50, T, 4)
    
    # Evaluator
    evaluator = RobustnessEvaluator(model, config)
    
    print("\nPerformance under different anomaly ratios:")
    print("-" * 40)
    
    anomaly_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = evaluator.evaluate_robustness(
        clean_X, fusion_target, anomaly_ratios
    )
    
    print("\nRobustness Summary:")
    print("-" * 40)
    baseline_mae = results[0.0]['mae']
    for ratio, res in results.items():
        degradation = (res['mae'] - baseline_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        print(f"  Anomaly ratio {ratio*100:4.0f}%: MAE={res['mae']:.4f} (degradation {degradation:+.1f}%)")


def main():
    print_banner()
    
    print("Select run mode:")
    print("  1. Model architecture only")
    print("  2. Data simulation demo")
    print("  3. Graph structure demo")
    print("  4. Quick training + inference")
    print("  5. Full pipeline (with robustness test)")
    
    choice = input("\nEnter choice (1-5, default 5): ").strip() or "5"
    
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
        # Full pipeline
        demo_model_architecture()
        demo_data_simulation()
        demo_graph_structure()
        model, trainer, config = run_training()
        demo_inference(model, config)
        run_robustness_test(model, config)
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
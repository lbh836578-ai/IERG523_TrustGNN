"""Generate larger synthetic datasets and quantitative summaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import get_agricultural_config
from data_simulator import AgriculturalDataSimulator, FaultInfo, GroundTruth, SensorDataTensor


OUTPUT_NAMES = ["temperature", "humidity", "soil_moisture", "light"]


def build_arrays(
    data_list: List[SensorDataTensor],
    gt_list: List[GroundTruth],
) -> Dict[str, np.ndarray]:
    """Stack generated samples into serializable NumPy arrays."""
    arrays = {
        "X": torch.stack([item.X for item in data_list]).cpu().numpy().astype(np.float32),
        "timestamps": np.stack([item.timestamps for item in data_list]).astype(np.int64),
        "fusion_target": torch.stack([item.fusion_target for item in gt_list]).cpu().numpy().astype(np.float32),
        "fault_mask": torch.stack([item.fault_mask for item in gt_list]).cpu().numpy().astype(np.float32),
        "fault_types": torch.stack([item.fault_types for item in gt_list]).cpu().numpy().astype(np.float32),
        "credibility_target": torch.stack([item.credibility_target for item in gt_list]).cpu().numpy().astype(np.float32),
        "sensor_ids": np.asarray(data_list[0].sensor_ids, dtype=str),
    }
    return arrays


def summarize_split(
    arrays: Dict[str, np.ndarray],
    fault_list: List[List[FaultInfo]],
) -> Dict[str, object]:
    """Create quantitative metrics for one generated split."""
    x = arrays["X"]
    fusion_target = arrays["fusion_target"]
    fault_mask = arrays["fault_mask"]
    credibility_target = arrays["credibility_target"]
    sensor_ids = arrays["sensor_ids"].tolist()

    fault_counter: Counter[str] = Counter(
        fault.fault_type.name for faults in fault_list for fault in faults
    )

    sensor_stats = {}
    for idx, sensor_id in enumerate(sensor_ids):
        sensor_values = x[:, idx, :, 0]
        sensor_fault_mask = fault_mask[:, idx, :]
        sensor_stats[sensor_id] = {
            "mean": float(sensor_values.mean()),
            "std": float(sensor_values.std()),
            "min": float(sensor_values.min()),
            "max": float(sensor_values.max()),
            "fault_ratio": float(sensor_fault_mask.mean()),
            "clean_ratio": float(1.0 - sensor_fault_mask.mean()),
        }

    output_stats = {}
    for idx, output_name in enumerate(OUTPUT_NAMES):
        output_values = fusion_target[:, :, idx]
        output_stats[output_name] = {
            "mean": float(output_values.mean()),
            "std": float(output_values.std()),
            "min": float(output_values.min()),
            "max": float(output_values.max()),
        }

    faulty_sample_count = sum(1 for faults in fault_list if faults)

    return {
        "sample_count": int(x.shape[0]),
        "num_sensors": int(x.shape[1]),
        "window_size": int(x.shape[2]),
        "feature_dim": int(x.shape[3]),
        "faulty_sample_count": int(faulty_sample_count),
        "faulty_sample_ratio": float(faulty_sample_count / max(x.shape[0], 1)),
        "fault_point_count": int(fault_mask.sum()),
        "fault_density": float(fault_mask.mean()),
        "avg_credibility_target": float(credibility_target.mean()),
        "fault_type_counts": dict(sorted(fault_counter.items())),
        "sensor_statistics": sensor_stats,
        "output_statistics": output_stats,
    }


def serialize_faults(fault_list: List[List[FaultInfo]]) -> List[Dict[str, object]]:
    """Convert fault metadata into JSON-serializable records."""
    records = []
    for sample_index, faults in enumerate(fault_list):
        records.append(
            {
                "sample_index": sample_index,
                "fault_count": len(faults),
                "faults": [
                    {
                        "sensor_id": fault.sensor_id,
                        "sensor_idx": fault.sensor_idx,
                        "fault_type": fault.fault_type.name,
                        "start_time": fault.start_time,
                        "end_time": fault.end_time,
                        "parameters": fault.parameters,
                    }
                    for fault in faults
                ],
            }
        )
    return records


def export_split(
    output_dir: Path,
    split_name: str,
    data_list: List[SensorDataTensor],
    gt_list: List[GroundTruth],
    fault_list: List[List[FaultInfo]],
) -> Dict[str, object]:
    """Export one dataset split as arrays plus summaries."""
    arrays = build_arrays(data_list, gt_list)
    summary = summarize_split(arrays, fault_list)
    fault_records = serialize_faults(fault_list)

    npz_path = output_dir / f"{split_name}.npz"
    summary_path = output_dir / f"{split_name}_summary.json"
    faults_path = output_dir / f"{split_name}_faults.json"

    np.savez_compressed(npz_path, **arrays)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    faults_path.write_text(json.dumps(fault_records, indent=2), encoding="utf-8")

    return {
        "files": {
            "dataset": npz_path.name,
            "summary": summary_path.name,
            "faults": faults_path.name,
        },
        "summary": summary,
    }


def generate_split(
    simulator: AgriculturalDataSimulator,
    sample_count: int,
    fault_ratio: float,
) -> Tuple[List[SensorDataTensor], List[GroundTruth], List[List[FaultInfo]]]:
    """Generate one synthetic split."""
    return simulator.generate_dataset(
        num_samples=sample_count,
        inject_faults=True,
        fault_ratio=fault_ratio,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for TrustFusion-GNN.")
    parser.add_argument("--output-dir", default="simulated_data", help="Directory for exported datasets.")
    parser.add_argument("--train-samples", type=int, default=3000, help="Number of training samples.")
    parser.add_argument("--val-samples", type=int, default=600, help="Number of validation samples.")
    parser.add_argument("--test-samples", type=int, default=600, help="Number of test samples.")
    parser.add_argument("--fault-ratio", type=float, default=0.35, help="Fault ratio used during generation.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_agricultural_config()
    split_specs = [
        ("train", args.train_samples, args.seed),
        ("val", args.val_samples, args.seed + 1),
        ("test", args.test_samples, args.seed + 2),
    ]

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "fault_ratio": args.fault_ratio,
        "base_seed": args.seed,
        "config": {
            "num_sensors": config.num_sensors,
            "window_size": config.window_size,
            "input_features": config.input_features,
            "output_features": config.output_features,
        },
        "splits": {},
    }

    for split_name, sample_count, seed in split_specs:
        simulator = AgriculturalDataSimulator(config, seed=seed)
        data_list, gt_list, fault_list = generate_split(
            simulator=simulator,
            sample_count=sample_count,
            fault_ratio=args.fault_ratio,
        )
        manifest["splits"][split_name] = export_split(
            output_dir=output_dir,
            split_name=split_name,
            data_list=data_list,
            gt_list=gt_list,
            fault_list=fault_list,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Synthetic datasets exported to {output_dir.resolve()}")
    print(f"Manifest written to {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Verify local VLM weights with one real training image and navigation prompt."""
import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Generated sft.json")
    parser.add_argument("--output", required=True, help="New JSON report path; never overwritten")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "config": str(Path(args.config).resolve()),
        "device": args.device,
        "seed": 42,
        "inference_inputs": ["current_front_camera", "legal_navigation_text"],
        "future_inputs": False,
    }
    # Reserve a new file before loading anything; preserve existing user reports.
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)

    started = time.perf_counter()
    torch = None
    device = None
    exit_code = 1
    try:
        repo = Path(__file__).resolve().parents[2]
        sys.path[:0] = [
            str(repo),
            str(repo / "train/verl"),
            str(repo / "train/verl/vla-adapter/openvla-oft"),
        ]
        import numpy as np
        import torch
        from navsim_rft.data import SceneDataset
        from navsim_rft.encoder import PrismaticEncoder

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("VLM startup verification requires an available CUDA device")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = json.loads(Path(args.config).read_text())
        if config.get("stage") != "sft":
            raise ValueError("Pass the generated SFT configuration (stage=sft)")
        dataset = SceneDataset(config["manifest"], config["stats"], training=True)
        if not len(dataset):
            raise ValueError("Training manifest contains no scenes")
        item = dataset[0]
        report.update(
            token=item["token"],
            manifest=str(Path(config["manifest"]).resolve()),
            vlm_path=str(Path(config["vlm"]).resolve()),
            image_size=list(item["image"].size),
            gpu_name=torch.cuda.get_device_name(device),
            torch_version=torch.__version__,
        )
        encoder = PrismaticEncoder(config["vlm"])
        report["loading_info"] = encoder.loading_info
        encoder.set_trainable(False)
        encoder = encoder.to(device).eval()
        report["parameter_count"] = sum(p.numel() for p in encoder.parameters())
        report["trainable_parameter_count"] = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        # Do not pass expert actions, future images, ego targets, or rewards.
        with torch.no_grad():
            features = encoder([item["image"]], [item["text"]])
        torch.cuda.synchronize(device)
        report["feature_shape"] = list(features.shape)
        report["feature_dtype"] = str(features.dtype)
        report["features_finite"] = bool(torch.isfinite(features).all().item())
        if not report["features_finite"]:
            raise FloatingPointError("VLM produced non-finite features")
        report["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(report["traceback"], file=sys.stderr)
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        if torch is not None and device is not None and device.type == "cuda":
            try:
                report["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            except Exception:
                report["peak_memory_bytes"] = None
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

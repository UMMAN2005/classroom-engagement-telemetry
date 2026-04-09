import types
import sys
import time
import warnings
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

if "_bz2" not in sys.modules:
    try:
        import _bz2  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("_bz2")
        _stub.BZ2Compressor = None  # type: ignore[attr-defined]
        _stub.BZ2Decompressor = None  # type: ignore[attr-defined]
        sys.modules["_bz2"] = _stub

import onnx
import onnxruntime as ort
from torchvision import models

warnings.filterwarnings(
    "ignore",
    message=".*legacy TorchScript-based ONNX.*",
)


NUM_WARMUP = 10
NUM_ITERATIONS = 100


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(weights_path: Path, num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(str(weights_path), map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    input_size: int,
) -> None:
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and validated: {onnx_path}")


def benchmark_pytorch(model: nn.Module, input_size: int) -> float:
    dummy = torch.randn(1, 3, input_size, input_size)

    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            model(dummy)

        start = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            model(dummy)
        elapsed = time.perf_counter() - start

    return (elapsed / NUM_ITERATIONS) * 1000


def benchmark_onnx(session, input_size: int) -> float:
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(
        1,
        3,
        input_size,
        input_size,
    ).astype(np.float32)

    for _ in range(NUM_WARMUP):
        session.run(None, {input_name: dummy})

    start = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        session.run(None, {input_name: dummy})
    elapsed = time.perf_counter() - start

    return (elapsed / NUM_ITERATIONS) * 1000


def main():
    project_root = Path(__file__).resolve().parents[2]
    config = load_config(project_root / "config.yaml")
    num_classes = len(config["classes"])
    input_size = config["model"]["classifier_input_size"]

    weights_path = project_root / "src" / "models" / "best_resnet18.pth"
    onnx_path = project_root / "src" / "models" / "model.onnx"

    print("Loading PyTorch model (ResNet18)...")
    model = build_model(weights_path, num_classes=num_classes)

    print("Exporting to ONNX...")
    export_onnx(model, onnx_path, input_size)

    pt_size_mb = weights_path.stat().st_size / (1024 * 1024)
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    print(
        f"\nBenchmarking PyTorch ({NUM_ITERATIONS} iterations, CPU)...",
    )
    pt_latency = benchmark_pytorch(model, input_size)

    print(
        f"Benchmarking ONNX Runtime ({NUM_ITERATIONS} iterations, CPU)...",
    )
    onnx_latency = benchmark_onnx(session, input_size)

    pt_fps = 1000 / pt_latency
    onnx_fps = 1000 / onnx_latency
    speedup = pt_latency / onnx_latency
    size_ratio = pt_size_mb / onnx_size_mb

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: dummy.numpy()})[0]
    parity_passed = bool(np.allclose(pt_out, onnx_out, atol=1e-5))
    max_diff = float(np.max(np.abs(pt_out - onnx_out)))

    print("\n## CPU Inference Benchmark (ResNet18)\n")
    print(
        f"| {'Runtime':<12} | {'Latency (ms/crop)':>18} "
        f"| {'FPS (crops/sec)':>16} "
        f"| {'Model Size (MB)':>16} |",
    )
    print(f"|{'-' * 14}|{'-' * 20}|{'-' * 18}|{'-' * 18}|")
    print(
        f"| {'PyTorch':<12} | {pt_latency:>18.2f} "
        f"| {pt_fps:>16.1f} | {pt_size_mb:>16.2f} |",
    )
    print(
        f"| {'ONNX (CPU)':<12} | {onnx_latency:>18.2f} "
        f"| {onnx_fps:>16.1f} "
        f"| {onnx_size_mb:>16.2f} |",
    )
    print(
        f"| {'Speedup':<12} | {speedup:>17.2f}x | {'':>16} | {size_ratio:>15.2f}x |",
    )
    print("\n## ONNX Parity Check\n")
    if parity_passed:
        print("ONNX Parity Check PASSED: outputs match PyTorch within 1e-5 tolerance.")
    else:
        print(f"ONNX Parity Check FAILED: max diff = {max_diff:.2e}")

    import csv

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "onnx_benchmark.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "runtime", "latency_ms", "fps", "model_size_mb",
            "speedup", "parity_passed", "max_diff",
        ])
        writer.writerow([
            "PyTorch", f"{pt_latency:.2f}", f"{pt_fps:.1f}",
            f"{pt_size_mb:.2f}", "", "", "",
        ])
        writer.writerow([
            "ONNX (CPU)", f"{onnx_latency:.2f}", f"{onnx_fps:.1f}",
            f"{onnx_size_mb:.2f}", f"{speedup:.2f}",
            parity_passed, f"{max_diff:.2e}",
        ])

    print(f"\nBenchmark results saved to {csv_path}")


if __name__ == "__main__":
    main()

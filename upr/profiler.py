import os
import time
import csv
import psutil
import torch
from typing import Dict, Any, Optional

class IsolatedTimer:
    """
    Fix 6 - Timer module to measure isolated phase durations independently.
    Phases: Conversion Time, Checkpoint Loading Time, Plane Loading Time,
    Reconstruction Time, Model Initialization Time, Forward Pass Time, Generation Time.
    Never reports a combined timing.
    """
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}

    def start(self, phase_name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._starts[phase_name] = time.perf_counter()

    def stop(self, phase_name: str) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = self._starts.pop(phase_name, time.perf_counter())
        elapsed = time.perf_counter() - start
        self.timers[phase_name] = elapsed
        return elapsed

    def get_summary(self) -> Dict[str, float]:
        return dict(self.timers)


class MemoryProfiler:
    """
    Phase 1.3 - Memory Profiler recording actual runtime usage (CPU RAM, GPU VRAM)
    separately from UPR representation metrics (planes loaded, bytes loaded, effective checkpoint size,
    compression ratio, bandwidth reduction) and theoretical future execution metrics.
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_cpu_ram_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def get_gpu_vram_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0

    def get_peak_gpu_vram_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        return 0.0

    def record_memory_snapshot(
        self,
        precision_bits: int,
        checkpoint_dir: str,
        output_csv_path: str = "results/memory.csv",
        total_planes: int = 16,
        fp16_vram_baseline_mb: float = 2926.21
    ) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(output_csv_path) if os.path.dirname(output_csv_path) else ".", exist_ok=True)
        
        cpu_ram = self.get_cpu_ram_mb()
        gpu_vram = self.get_gpu_vram_mb()
        peak_vram = self.get_peak_gpu_vram_mb()
        
        dir_size_bytes = 0
        if os.path.exists(checkpoint_dir):
            for root, _, files in os.walk(checkpoint_dir):
                for f in files:
                    dir_size_bytes += os.path.getsize(os.path.join(root, f))
                    
        full_checkpoint_size_mb = dir_size_bytes / (1024 * 1024)
        
        # Phase 1.3 Correct Accounting Computations
        planes_loaded = int(precision_bits)
        plane_fraction = planes_loaded / float(total_planes)
        
        bytes_loaded = int(dir_size_bytes * plane_fraction)
        effective_checkpoint_size_mb = round(full_checkpoint_size_mb * plane_fraction, 2)
        compression_ratio = round(plane_fraction, 4)
        bandwidth_reduction_pct = round((1.0 - plane_fraction) * 100.0, 2)
        
        # Theoretical future metrics (analytically calculated only)
        future_expected_vram_mb = round(fp16_vram_baseline_mb * plane_fraction, 2)
        future_expected_bandwidth_pct = round(plane_fraction * 100.0, 2)
        
        row = {
            "precision_bits": precision_bits,
            "planes_loaded": planes_loaded,
            "runtime_cpu_ram_mb": round(cpu_ram, 2),
            "runtime_gpu_vram_mb": round(gpu_vram, 2),
            "peak_gpu_vram_mb": round(peak_vram, 2),
            "full_checkpoint_size_mb": round(full_checkpoint_size_mb, 2),
            "bytes_loaded": bytes_loaded,
            "effective_checkpoint_size_mb": effective_checkpoint_size_mb,
            "representation_size_mb": effective_checkpoint_size_mb,
            "compression_ratio": compression_ratio,
            "bandwidth_reduction_pct": bandwidth_reduction_pct,
            "theoretical_future_vram_mb": future_expected_vram_mb,
            "theoretical_future_bandwidth_pct": future_expected_bandwidth_pct,
            # Backward-compatible fields
            "cpu_ram_mb": round(cpu_ram, 2),
            "gpu_vram_mb": round(gpu_vram, 2),
            "checkpoint_size_mb": round(full_checkpoint_size_mb, 2)
        }
        
        file_exists = os.path.exists(output_csv_path)
        with open(output_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
        return row


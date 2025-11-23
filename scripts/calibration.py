#!/usr/bin/env python3
"""
MUSE Calibration Module
Measures hardware capabilities and model characteristics to predict resource usage.
"""
import time
import gc
import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress

# Try to import model, handle error if not found (e.g. during initial setup check)
try:
    from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig
except ImportError:
    ConfigurableResNetBK = None

console = Console()

class MuseCalibrator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vram_total = 0
        self.memory_coeffs = {'base': 0, 'per_token': 0, 'per_param': 0}
        self.speed_coeffs = {'base': 0, 'per_token': 0}

        if self.device.type == 'cuda':
            self.vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2) # MB
        else:
            # Mock for CPU (system memory) - simplified
            import psutil
            self.vram_total = psutil.virtual_memory().total / (1024**2)

    def _clear_memory(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    def measure_run(self, config, batch_size, seq_len):
        if ConfigurableResNetBK is None:
            return 0, 0

        self._clear_memory()

        # Update config
        config.d_model = config.d_model
        config.n_layers = config.n_layers
        config.vocab_size = 1000 # Small vocab for calibration

        try:
            model = ConfigurableResNetBK(config).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters())

            # Dummy input
            x = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
            y = torch.randint(0, 1000, (batch_size * seq_len,)).to(self.device)

            # Measure Memory (Peak)
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            # Forward + Backward
            start_time = time.time()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()

            if self.device.type == 'cuda':
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            else:
                peak_mem = 0 # CPU accurate measurement is hard in Python, approximation needed

            del model, optimizer, x, y, logits, loss
            self._clear_memory()

            return peak_mem, end_time - start_time

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._clear_memory()
                return float('inf'), float('inf')
            raise e

    def calibrate(self):
        """Run calibration points to fit the model."""
        if ConfigurableResNetBK is None:
            return False

        console.print("[bold blue]Running System Calibration...[/bold blue]")

        points = [] # (N, B, D, L, Mem, Time)

        # Define probe points (conservative to avoid OOM during calibration)
        probes = [
            (128, 1, 128, 2),
            (512, 1, 128, 2),
            (128, 4, 128, 2),
            (128, 1, 256, 2),
            (128, 1, 128, 4),
        ]

        base_config = ResNetBKConfig(d_model=128, n_layers=2, vocab_size=1000, n_seq=128)

        with Progress() as progress:
            task = progress.add_task("[cyan]Measuring...", total=len(probes))

            for seq_len, batch, d_model, layers in probes:
                base_config.d_model = d_model
                base_config.n_layers = layers
                base_config.n_seq = seq_len

                mem, duration = self.measure_run(base_config, batch, seq_len)
                if mem != float('inf'):
                    # Approximate: Mem = Base + k * (B*N*D*L)
                    # This is a simplification of the O(N) MUSE formula
                    complexity = batch * seq_len * d_model * layers
                    points.append((complexity, mem, duration))

                progress.update(task, advance=1)

        if not points:
            console.print("[red]Calibration failed (all probes OOM).[/red]")
            return False

        # Simple Linear Regression for Memory
        # Mem = Base + Alpha * Complexity
        X = np.array([p[0] for p in points])
        Y_mem = np.array([p[1] for p in points])

        # A = [X, 1]
        A = np.vstack([X, np.ones(len(X))]).T
        alpha, base_mem = np.linalg.lstsq(A, Y_mem, rcond=None)[0]

        self.memory_coeffs['per_complex'] = alpha
        self.memory_coeffs['base'] = max(0, base_mem)

        # Throughput
        Y_time = np.array([p[2] for p in points])
        beta, base_time = np.linalg.lstsq(A, Y_time, rcond=None)[0]
        self.speed_coeffs['per_complex'] = beta
        self.speed_coeffs['base'] = max(0, base_time)

        console.print(f"Calibration Result: Alpha={alpha:.2e}, Beta={beta:.2e}")
        return True

    def predict(self, batch, seq_len, d_model, layers):
        complexity = batch * seq_len * d_model * layers

        mem_mb = self.memory_coeffs['base'] + self.memory_coeffs['per_complex'] * complexity
        # Add safety margin (optimizer states etc might scale differently)
        mem_mb *= 1.2

        time_sec = self.speed_coeffs['base'] + self.speed_coeffs['per_complex'] * complexity

        return mem_mb, time_sec

    def check_safety(self, mem_mb):
        return mem_mb < (self.vram_total * 0.9)

if __name__ == "__main__":
    cal = MuseCalibrator()
    if cal.device.type == 'cuda':
        cal.calibrate()
        # Test Prediction
        m, t = cal.predict(4, 1024, 512, 6)
        console.print(f"Prediction for B=4, N=1024, D=512, L=6:")
        console.print(f"  Memory: {m:.1f} MB (Limit: {cal.vram_total:.1f} MB)")
        console.print(f"  Step Time: {t*1000:.1f} ms")
    else:
        console.print("Skipping calibration (CPU mode)")

#!/usr/bin/env python3
"""
MUSE Calibration Module
Measures hardware capabilities and model characteristics to predict resource usage.
"""
import time
import sys
import gc
import torch
import numpy as np
import warnings
from rich.console import Console
from rich.progress import Progress

# Try to import model, handle error if not found (e.g. during initial setup check)
try:
    from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig
    from src.models.bk_core import set_triton_mode
except ImportError:
    ConfigurableResNetBK = None
    def set_triton_mode(x): pass

console = Console()

class MuseCalibrator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_triton = False
        try:
            import triton
            self.has_triton = True
        except ImportError:
            self.has_triton = False

        self.vram_total = 0
        self.memory_coeffs = {'base': 0, 'per_token': 0, 'per_complex': 0}
        self.speed_coeffs = {'base': 0, 'per_token': 0, 'per_complex': 0}

        if self.device.type == 'cuda':
            try:
                self.vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2) # MB
                # Enforce Triton Mode for calibration on GPU if available
                if self.has_triton:
                    set_triton_mode(True)
            except:
                self.vram_total = 0
        else:
            # Mock for CPU (system memory) - simplified
            try:
                import psutil
                self.vram_total = psutil.virtual_memory().total / (1024**2)
            except:
                self.vram_total = 0

    def _clear_memory(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()

    def estimate_exact(self, batch_size, seq_len, d_model, n_layers, vocab_size=30000):
        """
        Run a single forward/backward to measure peak memory and step time.
        This is slower but more accurate than linear regression.
        Returns: (peak_mem_mb, step_time_sec) or (None, None) if failed/OOM
        """
        if ConfigurableResNetBK is None or self.device.type != 'cuda':
            return None, None

        cfg = ResNetBKConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_seq=seq_len,
            vocab_size=vocab_size,
        )

        try:
            self._clear_memory()
            model = ConfigurableResNetBK(cfg).to(self.device)
            opt = torch.optim.AdamW(model.parameters())

            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            y = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)

            # warmup
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out.view(-1, out.size(-1)), y)
            loss.backward()
            opt.zero_grad()

            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start = time.time()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(out.view(-1, out.size(-1)), y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            step_time = end - start

            del model, opt, x, y, out, loss
            self._clear_memory()
            return peak_mem, step_time
        except Exception:
            # Catch all exceptions (OOM, Triton Failure)
            self._clear_memory()
            return None, None

    def measure_run(self, config, batch_size, seq_len):
        if ConfigurableResNetBK is None:
            return 0, 0

        self._clear_memory()

        # Update config
        config.d_model = config.d_model
        config.n_layers = config.n_layers
        config.vocab_size = 1000 # Small vocab for calibration

        try:
            # Suppress Triton warnings during calibration
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                model = ConfigurableResNetBK(config).to(self.device)
                optimizer = torch.optim.AdamW(model.parameters())

                # Dummy input
                x = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
                y = torch.randint(0, 1000, (batch_size * seq_len,)).to(self.device)

                # Warmup (1 step) to settle allocations/caches
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y)
                loss.backward()
                optimizer.zero_grad()
                del logits, loss

                # Reset stats after warmup
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                # Measurement Run (Forward + Backward)
                start_time = time.time()
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()

                if self.device.type == 'cuda':
                    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                else:
                    peak_mem = 0 # CPU accurate measurement is hard in Python

                del model, optimizer, x, y, logits, loss
                self._clear_memory()

                return peak_mem, end_time - start_time

        except Exception as e:
            # Catch RuntimeErrors (OOM) and ValueErrors (Triton issues)
            # Log minimal info if needed, but don't crash
            # if "out of memory" in str(e).lower():
            #     return float('inf'), float('inf')
            # For any error (OOM or Kernel failure), treat as invalid point
            self._clear_memory()
            return float('inf'), float('inf')

    def calibrate(self):
        """Run calibration points to fit the model."""
        if ConfigurableResNetBK is None:
            return False

        # Use rich status or print
        # console.print("[bold blue]Running System Calibration...[/bold blue]")

        # Initial Probes
        probes = [
            (128, 1, 128, 2),
            (256, 1, 128, 2),
            (128, 2, 128, 2),
            (128, 1, 256, 2),
            (128, 1, 128, 4),
            (256, 2, 256, 2), # Heavier point
        ]

        # Heavy Probes for Deep Scan
        heavy_probes = [
            (512, 2, 256, 4),
            (512, 4, 256, 4),
            (1024, 2, 256, 4),
            (512, 2, 512, 4),
            (512, 2, 256, 8),
        ]

        def run_probes(probe_list, label="Measuring..."):
            measured_points = []
            base_config = ResNetBKConfig(d_model=128, n_layers=2, vocab_size=1000, n_seq=128)
            # Use Progress
            # Note: Progress bar handling moved to outer scope/wizard usually.
            # But for standalone calibration, we keep it here.
            # We assume this is running under status/progress in wizard if called from there?
            # Actually, `calibrate()` in wizard was just `cal.calibrate()`.
            # We can use a context manager to suppress output if needed, but `calibrate` prints to console.
            # Let's keep it clean.

            with Progress() as progress:
                task = progress.add_task(f"[cyan]{label}", total=len(probe_list))
                for seq_len, batch, d_model, layers in probe_list:
                    base_config.d_model = d_model
                    base_config.n_layers = layers
                    base_config.n_seq = seq_len
                    mem, duration = self.measure_run(base_config, batch, seq_len)
                    if mem != float('inf') and mem > 0:
                        complexity = batch * seq_len * d_model * layers
                        measured_points.append((complexity, mem, duration))
                    progress.update(task, advance=1)
            return measured_points

        points = run_probes(probes)

        # Helper to compute coefficients
        def compute_regression(pts):
            if len(pts) < 2: return -1, -1, -1, -1
            X = np.array([p[0] for p in pts])
            Y_mem = np.array([p[1] for p in pts])
            Y_time = np.array([p[2] for p in pts])
            A = np.vstack([X, np.ones(len(X))]).T
            try:
                a, b_mem = np.linalg.lstsq(A, Y_mem, rcond=None)[0]
                b, b_time = np.linalg.lstsq(A, Y_time, rcond=None)[0]
                return a, b_mem, b, b_time
            except:
                return -1, -1, -1, -1

        alpha, base_mem, beta, base_time = compute_regression(points)

        # Check for invalid results (negative slope/intercept)
        if alpha <= 0 or base_mem < 0:
            console.print("[yellow]Initial calibration noisy (Negative Slope/Base). Switching to Deep Scan...[/yellow]")
            # Retry with Heavy Probes
            points = run_probes(heavy_probes, label="Deep Scan...")
            alpha, base_mem, beta, base_time = compute_regression(points)

        # Final Decision
        use_fallback = False
        if alpha <= 0 or base_mem < 0:
             console.print(f"[red]Deep Scan Failed or Noisy (Alpha={alpha:.2e}). Fallback to theoretical model.[/red]")
             use_fallback = True
        else:
             self.memory_coeffs['per_complex'] = alpha
             self.memory_coeffs['base'] = base_mem
             self.speed_coeffs['per_complex'] = beta
             self.speed_coeffs['base'] = max(0, base_time)
             console.print(f"Calibration Result: Alpha={alpha:.2e}, Beta={beta:.2e}")

        if use_fallback:
            # Theoretical Fallback (Last Resort)
            self.memory_coeffs['per_complex'] = 2.0e-5 # slightly conservative
            self.memory_coeffs['base'] = 500 # 500MB fixed overhead
            self.speed_coeffs['per_complex'] = 1.0e-8
            self.speed_coeffs['base'] = 0.0

        return True

    def predict(self, batch, seq_len, d_model, layers):
        complexity = batch * seq_len * d_model * layers

        # Model Weights (Static Memory) - explicit calculation
        param_count = 12 * layers * (d_model**2) + (50000 * d_model)
        static_mem_mb = (param_count * 4) / (1024**2)

        # Dynamic Memory (Activations)
        dynamic_mem_mb = self.memory_coeffs['per_complex'] * complexity

        # Total
        mem_mb = self.memory_coeffs['base'] + static_mem_mb + dynamic_mem_mb

        # Add safety margin
        mem_mb *= 1.1

        time_sec = self.speed_coeffs['base'] + self.speed_coeffs['per_complex'] * complexity

        return mem_mb, time_sec

    def check_safety(self, mem_mb):
        if self.vram_total == 0: return True # Cannot check
        return mem_mb < (self.vram_total * 0.9)

    def check_triton(self, strict=True):
        """Check if Triton is available. Exit if strict=True and missing."""
        if not self.has_triton:
            if strict:
                console.print("[bold red]FATAL: Triton is missing or failed to load.[/bold red]")
                console.print("This project requires Triton for O(N) performance on GPU.")
                console.print("Please install triton or run in a compatible environment (Linux/WSL).")
                sys.exit(1)
            else:
                return False
        return True

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

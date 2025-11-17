# PowerShell version of run_all_paper_experiments.sh
# Run all experiments needed for paper

Write-Host "=== Running All Paper Experiments ===" -ForegroundColor Green
Write-Host "This will take approximately 24-48 hours on 4x T4 GPUs"
Write-Host ""

# Create results directory
New-Item -ItemType Directory -Path "results/paper_experiments" -Force | Out-Null

# 1. Long-Context Stability Experiments
Write-Host "[1/5] Running long-context stability experiments..." -ForegroundColor Cyan
python scripts/benchmarks/run_scaling_experiments.py `
    --model resnet_bk `
    --seq_lengths 8192,32768,131072,524288,1048576 `
    --seeds 42,43,44,45,46 `
    --output results/paper_experiments/long_context_resnet_bk.json

python scripts/benchmarks/run_scaling_experiments.py `
    --model mamba `
    --seq_lengths 8192,32768,131072 `
    --seeds 42,43,44,45,46 `
    --output results/paper_experiments/long_context_mamba.json

# 2. Quantization Robustness Experiments
Write-Host "[2/5] Running quantization experiments..." -ForegroundColor Cyan
python scripts/benchmarks/run_quantization_sweep.py `
    --model resnet_bk `
    --bits FP32,FP16,INT8,INT4 `
    --seeds 42,43,44,45,46 `
    --output results/paper_experiments/quantization_resnet_bk.json

python scripts/benchmarks/run_quantization_sweep.py `
    --model mamba `
    --bits FP32,FP16,INT8,INT4 `
    --seeds 42,43,44,45,46 `
    --output results/paper_experiments/quantization_mamba.json

# 3. Dynamic Efficiency Experiments
Write-Host "[3/5] Running efficiency experiments..." -ForegroundColor Cyan
python scripts/benchmarks/measure_flops.py `
    --models resnet_bk,resnet_bk_act,mamba `
    --seq_length 2048 `
    --seeds 42,43,44,45,46 `
    --output results/paper_experiments/efficiency.json

# 4. Ablation Studies
Write-Host "[4/5] Running ablation studies..." -ForegroundColor Cyan
python scripts/benchmarks/run_ablation.py `
    --components prime_bump,scattering_router,lap_stability,semiseparable `
    --seeds 42,43,44,45,46 `
    --output results/paper_experiments/ablation.json

# 5. Generate Figures
Write-Host "[5/5] Generating figures..." -ForegroundColor Cyan
python scripts/benchmarks/generate_stability_graph.py `
    --input "results/paper_experiments/long_context_*.json" `
    --output results/paper_experiments/figure1_stability.pdf

python scripts/benchmarks/generate_quantization_graph.py `
    --input "results/paper_experiments/quantization_*.json" `
    --output results/paper_experiments/figure2_quantization.pdf

python scripts/benchmarks/generate_efficiency_graph.py `
    --input results/paper_experiments/efficiency.json `
    --output results/paper_experiments/figure3_efficiency.pdf

Write-Host ""
Write-Host "=== All Experiments Complete ===" -ForegroundColor Green
Write-Host "Results saved to: results/paper_experiments/"
Write-Host "Figures saved to: results/paper_experiments/figure*.pdf"

"""
CUDA VRAMエミュレーションを設定するユーティリティ。

- torch.cuda.set_per_process_memory_fractionを用いて8GBターゲットの割合を設定
- 実行結果をJSON/コンソール両方に出力

注意: ソフトリミットのため、厳密なcgroup制限は別途必要。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.physics_checks import ensure_results_dir, set_cuda_vram_limit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gb", type=float, default=8.0, help="Desired VRAM cap in GB.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/infra/vram_emulation.json"),
        help="Path to save JSON result.",
    )
    args = parser.parse_args()

    info = set_cuda_vram_limit(args.target_gb)
    ensure_results_dir(args.output)
    args.output.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"[done] Soft VRAM cap set for target {args.target_gb} GB")


if __name__ == "__main__":
    main()

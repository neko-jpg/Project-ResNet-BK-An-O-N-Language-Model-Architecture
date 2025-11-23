import json
import os
import re

def main():
    json_path = "results/ci_bench/local_efficiency_results.json"
    tex_path = "paper/main.tex"

    if not os.path.exists(json_path):
        print(f"No benchmark results found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Format table
    rows = []
    for model_name, metrics in data.items():
        if model_name == "config": continue
        flops = metrics.get('avg_flops', 0) / 1e9
        loss = metrics.get('avg_loss', 0)
        model_display = model_name.replace("_", " ").title()
        rows.append(f"{model_display} & {loss:.4f} & {flops:.2f}G \\\\")

    table_content = r"""
\begin{table}[ht]
\centering
\caption{Automated CI Benchmark Results (Lightweight)}
\label{tab:ci_bench}
\begin{tabular}{lcc}
\toprule
Model & Loss & FLOPs (G) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    with open(tex_path, 'r') as f:
        content = f.read()

    # Insert before \end{document}
    marker = "\\section{Automated Benchmarks}"
    if marker in content:
         print(f"'{marker}' already exists in {tex_path}. Please update manually or delete the section to regenerate.")
    else:
        insert_point = content.rfind("\\end{document}")
        if insert_point != -1:
            new_content = content[:insert_point] + "\n" + marker + "\n" + table_content + "\n" + content[insert_point:]
            with open(tex_path, 'w') as f:
                f.write(new_content)
            print(f"Updated {tex_path} with benchmark results.")
        else:
            print("Could not find \\end{document} tag.")

if __name__ == "__main__":
    main()

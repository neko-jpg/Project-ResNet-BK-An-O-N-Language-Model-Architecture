"""
Generate Dynamic Efficiency Graph

Creates publication-quality visualization comparing ResNet-BK and Mamba
dynamic compute efficiency: PPL vs average FLOPs per token.

Implements Task 18 from mamba-killer-ultra-scale spec.

Requirements: 8.9, 8.10, 8.11, 8.12

Usage:
    # Generate from existing results
    python scripts/generate_efficiency_graph.py --results_dir results/efficiency
    
    # Generate with simulated data (for testing)
    python scripts/generate_efficiency_graph.py --simulate
    
    # Customize output
    python scripts/generate_efficiency_graph.py --output efficiency_graph.pdf --dpi 300
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8


def load_efficiency_results(results_dir: Path, model_name: str) -> Dict[str, Dict]:
    """
    Load efficiency results from JSON files.
    
    Args:
        results_dir: directory containing results
        model_name: 'resnetbk' or 'mamba'
    
    Returns:
        Dictionary mapping configuration to efficiency metrics
    """
    results = {}
    
    # Look for results files
    pattern = f"{model_name}_efficiency*.json"
    result_files = list(results_dir.glob(pattern))
    
    if not result_files:
        # Try alternative naming
        pattern = f"efficiency_{model_name}_results.json"
        result_files = list(results_dir.glob(pattern))
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract configuration identifier
        config_id = data.get('config_id', result_file.stem)
        results[config_id] = data
    
    return results


def simulate_efficiency_results(
    model_type: str,
    num_configs: int = 10,
    seed: int = 42
) -> List[Dict]:
    """
    Simulate efficiency results for testing.
    
    Generates data points showing PPL vs FLOPs trade-off for different
    configurations (ACT thresholds, sparsity levels, etc.)
    
    Args:
        model_type: 'resnetbk' or 'mamba'
        num_configs: number of configurations to simulate
        seed: random seed
    
    Returns:
        List of dictionaries with efficiency metrics
    """
    np.random.seed(seed)
    results = []
    
    if model_type == 'resnetbk':
        # ResNet-BK: Efficient across range
        # Requirement 8.10: Achieves PPL=30 with 2× fewer FLOPs than Mamba
        
        # Base configuration (no optimizations)
        base_flops = 5.0e9  # 5 GFLOPs per token
        base_ppl = 28.0
        
        # Generate configurations with varying compute budgets
        for i in range(num_configs):
            # Compute reduction factor (1.0 = full, 0.4 = 60% reduction)
            compute_factor = 1.0 - (i / num_configs) * 0.6
            
            # FLOPs scale linearly with compute
            flops_per_token = base_flops * compute_factor
            
            # PPL increases slowly with compute reduction (robust)
            # At 40% FLOPs reduction, PPL increases by ~5%
            ppl_increase = (1.0 - compute_factor) * 0.15
            ppl = base_ppl * (1.0 + ppl_increase)
            
            # Add small noise
            ppl += np.random.normal(0, 0.3)
            flops_per_token += np.random.normal(0, 0.1e9)
            
            results.append({
                'config_id': f'resnetbk_config_{i}',
                'model': 'resnetbk',
                'perplexity': max(ppl, 20.0),
                'flops_per_token': max(flops_per_token, 1.0e9),
                'compute_factor': compute_factor,
            })
    
    else:  # mamba
        # Mamba: Less efficient, worse PPL-FLOPs trade-off
        # Requirement 8.11: At equal FLOPs, ResNet-BK has 30% lower PPL
        
        # Base configuration
        base_flops = 10.0e9  # 10 GFLOPs per token (2× ResNet-BK)
        base_ppl = 32.0  # Slightly worse baseline
        
        # Generate configurations
        for i in range(num_configs):
            compute_factor = 1.0 - (i / num_configs) * 0.5  # Less aggressive reduction
            
            # FLOPs scale
            flops_per_token = base_flops * compute_factor
            
            # PPL increases more rapidly with compute reduction (less robust)
            ppl_increase = (1.0 - compute_factor) * 0.35  # 2× worse than ResNet-BK
            ppl = base_ppl * (1.0 + ppl_increase)
            
            # Add noise
            ppl += np.random.normal(0, 0.5)
            flops_per_token += np.random.normal(0, 0.2e9)
            
            results.append({
                'config_id': f'mamba_config_{i}',
                'model': 'mamba',
                'perplexity': max(ppl, 25.0),
                'flops_per_token': max(flops_per_token, 2.0e9),
                'compute_factor': compute_factor,
            })
    
    return results



def plot_efficiency_graph(
    resnetbk_results: List[Dict],
    mamba_results: List[Dict],
    output_path: str,
    dpi: int = 300,
    format: str = 'pdf'
):
    """
    Generate publication-quality efficiency comparison graph.
    
    Requirements:
    - 8.9: Plot PPL vs average FLOPs per token
    - 8.10: Show ResNet-BK achieving PPL=30 with 2× fewer FLOPs than Mamba
    - 8.11: Annotate "Pareto frontier" showing ResNet-BK dominance
    - 8.12: Generate publication-quality figure
    
    Args:
        resnetbk_results: ResNet-BK efficiency results
        mamba_results: Mamba efficiency results
        output_path: output file path
        dpi: resolution for raster formats
        format: output format (pdf, svg, png, eps)
    """
    # Create figure with appropriate size for publication
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color scheme
    resnetbk_color = '#2E86AB'  # Blue
    mamba_color = '#A23B72'     # Red/Purple
    pareto_color = '#06A77D'    # Green
    target_color = '#F18F01'    # Orange
    
    # Extract data
    rb_flops = np.array([r['flops_per_token'] / 1e9 for r in resnetbk_results])  # Convert to GFLOPs
    rb_ppl = np.array([r['perplexity'] for r in resnetbk_results])
    
    mb_flops = np.array([r['flops_per_token'] / 1e9 for r in mamba_results])
    mb_ppl = np.array([r['perplexity'] for r in mamba_results])
    
    # Sort by FLOPs for line plotting
    rb_sort_idx = np.argsort(rb_flops)
    rb_flops_sorted = rb_flops[rb_sort_idx]
    rb_ppl_sorted = rb_ppl[rb_sort_idx]
    
    mb_sort_idx = np.argsort(mb_flops)
    mb_flops_sorted = mb_flops[mb_sort_idx]
    mb_ppl_sorted = mb_ppl[mb_sort_idx]
    
    # Plot ResNet-BK
    ax.plot(
        rb_flops_sorted,
        rb_ppl_sorted,
        color=resnetbk_color,
        marker='o',
        markersize=10,
        linewidth=3,
        label='ResNet-BK',
        zorder=3,
        alpha=0.8
    )
    
    # Plot Mamba
    ax.plot(
        mb_flops_sorted,
        mb_ppl_sorted,
        color=mamba_color,
        marker='s',
        markersize=10,
        linewidth=3,
        label='Mamba',
        zorder=3,
        alpha=0.8
    )
    
    # Highlight key points
    # Requirement 8.10: ResNet-BK achieving PPL=30 with 2× fewer FLOPs
    
    # Find ResNet-BK point closest to PPL=30
    rb_target_idx = np.argmin(np.abs(rb_ppl - 30.0))
    rb_target_flops = rb_flops[rb_target_idx]
    rb_target_ppl = rb_ppl[rb_target_idx]
    
    # Find Mamba point with similar PPL
    mb_similar_ppl_idx = np.argmin(np.abs(mb_ppl - rb_target_ppl))
    mb_similar_flops = mb_flops[mb_similar_ppl_idx]
    mb_similar_ppl = mb_ppl[mb_similar_ppl_idx]
    
    # Highlight ResNet-BK target point
    ax.scatter(
        [rb_target_flops],
        [rb_target_ppl],
        s=400,
        color=target_color,
        marker='*',
        edgecolors='black',
        linewidths=2,
        zorder=5,
        label=f'ResNet-BK target (PPL≈30)'
    )
    
    # Annotate ResNet-BK target
    ax.annotate(
        f'ResNet-BK\nPPL = {rb_target_ppl:.1f}\nFLOPs = {rb_target_flops:.2f} G',
        xy=(rb_target_flops, rb_target_ppl),
        xytext=(rb_target_flops - 1.5, rb_target_ppl + 3),
        fontsize=10,
        color=resnetbk_color,
        fontweight='bold',
        arrowprops=dict(
            arrowstyle='->',
            color=resnetbk_color,
            lw=2.5,
            connectionstyle='arc3,rad=0.3'
        ),
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor=resnetbk_color,
            linewidth=2,
            alpha=0.95
        )
    )
    
    # Annotate Mamba comparison point
    if mb_similar_flops > rb_target_flops * 1.5:  # If significantly more FLOPs
        ax.scatter(
            [mb_similar_flops],
            [mb_similar_ppl],
            s=300,
            color='red',
            marker='X',
            edgecolors='black',
            linewidths=2,
            zorder=5
        )
        
        ax.annotate(
            f'Mamba (similar PPL)\nPPL = {mb_similar_ppl:.1f}\nFLOPs = {mb_similar_flops:.2f} G\n({mb_similar_flops/rb_target_flops:.1f}× more FLOPs)',
            xy=(mb_similar_flops, mb_similar_ppl),
            xytext=(mb_similar_flops + 1.0, mb_similar_ppl + 3),
            fontsize=10,
            color=mamba_color,
            fontweight='bold',
            arrowprops=dict(
                arrowstyle='->',
                color=mamba_color,
                lw=2.5,
                connectionstyle='arc3,rad=-0.3'
            ),
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor=mamba_color,
                linewidth=2,
                alpha=0.95
            )
        )
    
    # Draw Pareto frontier
    # Requirement 8.11: Annotate "Pareto frontier" showing ResNet-BK dominance
    
    # Compute Pareto frontier for ResNet-BK
    # A point is on the Pareto frontier if no other point has both lower FLOPs and lower PPL
    rb_pareto_mask = np.ones(len(rb_flops), dtype=bool)
    for i in range(len(rb_flops)):
        for j in range(len(rb_flops)):
            if i != j:
                # If point j dominates point i (lower FLOPs AND lower PPL)
                if rb_flops[j] < rb_flops[i] and rb_ppl[j] < rb_ppl[i]:
                    rb_pareto_mask[i] = False
                    break
    
    rb_pareto_flops = rb_flops[rb_pareto_mask]
    rb_pareto_ppl = rb_ppl[rb_pareto_mask]
    
    # Sort Pareto points by FLOPs
    pareto_sort_idx = np.argsort(rb_pareto_flops)
    rb_pareto_flops_sorted = rb_pareto_flops[pareto_sort_idx]
    rb_pareto_ppl_sorted = rb_pareto_ppl[pareto_sort_idx]
    
    # Draw Pareto frontier line
    ax.plot(
        rb_pareto_flops_sorted,
        rb_pareto_ppl_sorted,
        color=pareto_color,
        linestyle='--',
        linewidth=3,
        label='ResNet-BK Pareto frontier',
        zorder=4,
        alpha=0.7
    )
    
    # Shade region dominated by ResNet-BK
    # Create polygon for shaded region
    if len(rb_pareto_flops_sorted) > 1:
        # Extend to axes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Create polygon vertices
        poly_x = [xlim[0]] + list(rb_pareto_flops_sorted) + [xlim[1], xlim[1], xlim[0]]
        poly_y = [ylim[1]] + list(rb_pareto_ppl_sorted) + [rb_pareto_ppl_sorted[-1], ylim[1], ylim[1]]
        
        ax.fill(
            poly_x,
            poly_y,
            color=pareto_color,
            alpha=0.1,
            zorder=1
        )
        
        # Add text in dominated region
        mid_x = np.mean(rb_pareto_flops_sorted)
        mid_y = np.mean(rb_pareto_ppl_sorted) + 2
        
        ax.text(
            mid_x,
            mid_y,
            'ResNet-BK\nDominates',
            ha='center',
            va='center',
            fontsize=14,
            color=pareto_color,
            fontweight='bold',
            alpha=0.6,
            style='italic',
            bbox=dict(
                boxstyle='round,pad=0.7',
                facecolor='white',
                edgecolor=pareto_color,
                linewidth=2,
                alpha=0.8
            )
        )
    
    # Formatting
    ax.set_xlabel('Average FLOPs per Token (GFLOPs)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Perplexity (lower is better)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Dynamic Compute Efficiency: ResNet-BK vs Mamba\nPerplexity vs FLOPs Trade-off',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Set axis limits with some padding
    all_flops = np.concatenate([rb_flops, mb_flops])
    all_ppl = np.concatenate([rb_ppl, mb_ppl])
    
    flops_range = all_flops.max() - all_flops.min()
    ppl_range = all_ppl.max() - all_ppl.min()
    
    ax.set_xlim(
        max(0, all_flops.min() - 0.1 * flops_range),
        all_flops.max() + 0.1 * flops_range
    )
    ax.set_ylim(
        max(0, all_ppl.min() - 0.1 * ppl_range),
        all_ppl.max() + 0.1 * ppl_range
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(
        loc='upper right',
        framealpha=0.95,
        edgecolor='black',
        fontsize=11,
        title='Model & Frontier',
        title_fontsize=12
    )
    
    # Add efficiency metrics text box
    # Calculate key metrics
    rb_best_ppl = rb_ppl.min()
    rb_best_flops = rb_flops[np.argmin(rb_ppl)]
    mb_best_ppl = mb_ppl.min()
    mb_best_flops = mb_flops[np.argmin(mb_ppl)]
    
    # Find equal PPL comparison
    target_ppl = 30.0
    rb_at_target = rb_flops[np.argmin(np.abs(rb_ppl - target_ppl))]
    mb_at_target = mb_flops[np.argmin(np.abs(mb_ppl - target_ppl))]
    flops_ratio = mb_at_target / rb_at_target if rb_at_target > 0 else 0
    
    # Find equal FLOPs comparison
    target_flops = 5.0  # GFLOPs
    rb_ppl_at_flops = rb_ppl[np.argmin(np.abs(rb_flops - target_flops))]
    mb_ppl_at_flops = mb_ppl[np.argmin(np.abs(mb_flops - target_flops))]
    ppl_improvement = ((mb_ppl_at_flops - rb_ppl_at_flops) / mb_ppl_at_flops) * 100
    
    metrics_text = (
        f"Key Metrics:\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"At PPL ≈ {target_ppl:.0f}:\n"
        f"  ResNet-BK: {rb_at_target:.2f} GFLOPs\n"
        f"  Mamba: {mb_at_target:.2f} GFLOPs\n"
        f"  Speedup: {flops_ratio:.1f}×\n"
        f"\n"
        f"At {target_flops:.1f} GFLOPs:\n"
        f"  ResNet-BK: PPL {rb_ppl_at_flops:.1f}\n"
        f"  Mamba: PPL {mb_ppl_at_flops:.1f}\n"
        f"  Improvement: {ppl_improvement:.1f}%"
    )
    
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.7',
            facecolor='white',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.95
        ),
        family='monospace'
    )
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path_obj = Path(output_path)
    
    # Save in specified format
    if format == 'pdf':
        plt.savefig(output_path_obj.with_suffix('.pdf'), dpi=dpi, bbox_inches='tight', format='pdf')
        print(f"✓ Saved PDF: {output_path_obj.with_suffix('.pdf')}")
    elif format == 'svg':
        plt.savefig(output_path_obj.with_suffix('.svg'), dpi=dpi, bbox_inches='tight', format='svg')
        print(f"✓ Saved SVG: {output_path_obj.with_suffix('.svg')}")
    elif format == 'eps':
        plt.savefig(output_path_obj.with_suffix('.eps'), dpi=dpi, bbox_inches='tight', format='eps')
        print(f"✓ Saved EPS: {output_path_obj.with_suffix('.eps')}")
    else:  # png
        plt.savefig(output_path_obj.with_suffix('.png'), dpi=dpi, bbox_inches='tight', format='png')
        print(f"✓ Saved PNG: {output_path_obj.with_suffix('.png')}")
    
    # Also save in PNG format for easy viewing
    if format != 'png':
        plt.savefig(output_path_obj.with_suffix('.png'), dpi=dpi, bbox_inches='tight', format='png')
        print(f"✓ Saved PNG: {output_path_obj.with_suffix('.png')}")
    
    plt.close()



def generate_summary_statistics(
    resnetbk_results: List[Dict],
    mamba_results: List[Dict]
) -> Dict:
    """
    Generate summary statistics for the comparison.
    
    Args:
        resnetbk_results: ResNet-BK efficiency results
        mamba_results: Mamba efficiency results
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'resnetbk': {},
        'mamba': {},
        'comparison': {}
    }
    
    # Extract arrays
    rb_flops = np.array([r['flops_per_token'] for r in resnetbk_results])
    rb_ppl = np.array([r['perplexity'] for r in resnetbk_results])
    
    mb_flops = np.array([r['flops_per_token'] for r in mamba_results])
    mb_ppl = np.array([r['perplexity'] for r in mamba_results])
    
    # ResNet-BK statistics
    summary['resnetbk'] = {
        'best_ppl': float(rb_ppl.min()),
        'best_ppl_flops': float(rb_flops[np.argmin(rb_ppl)]),
        'lowest_flops': float(rb_flops.min()),
        'lowest_flops_ppl': float(rb_ppl[np.argmin(rb_flops)]),
        'mean_ppl': float(rb_ppl.mean()),
        'mean_flops': float(rb_flops.mean()),
        'num_configs': len(resnetbk_results),
    }
    
    # Mamba statistics
    summary['mamba'] = {
        'best_ppl': float(mb_ppl.min()),
        'best_ppl_flops': float(mb_flops[np.argmin(mb_ppl)]),
        'lowest_flops': float(mb_flops.min()),
        'lowest_flops_ppl': float(mb_ppl[np.argmin(mb_flops)]),
        'mean_ppl': float(mb_ppl.mean()),
        'mean_flops': float(mb_flops.mean()),
        'num_configs': len(mamba_results),
    }
    
    # Comparison at equal PPL
    target_ppls = [25.0, 30.0, 35.0, 40.0]
    equal_ppl_comparisons = {}
    
    for target_ppl in target_ppls:
        rb_idx = np.argmin(np.abs(rb_ppl - target_ppl))
        mb_idx = np.argmin(np.abs(mb_ppl - target_ppl))
        
        rb_flops_at_ppl = rb_flops[rb_idx]
        mb_flops_at_ppl = mb_flops[mb_idx]
        
        if rb_flops_at_ppl > 0:
            speedup = mb_flops_at_ppl / rb_flops_at_ppl
        else:
            speedup = 0
        
        equal_ppl_comparisons[f'ppl_{int(target_ppl)}'] = {
            'target_ppl': target_ppl,
            'resnetbk_actual_ppl': float(rb_ppl[rb_idx]),
            'mamba_actual_ppl': float(mb_ppl[mb_idx]),
            'resnetbk_flops': float(rb_flops_at_ppl),
            'mamba_flops': float(mb_flops_at_ppl),
            'flops_ratio': float(speedup),
        }
    
    summary['comparison']['equal_ppl'] = equal_ppl_comparisons
    
    # Comparison at equal FLOPs
    target_flops_list = [3e9, 5e9, 7e9, 10e9]
    equal_flops_comparisons = {}
    
    for target_flops in target_flops_list:
        rb_idx = np.argmin(np.abs(rb_flops - target_flops))
        mb_idx = np.argmin(np.abs(mb_flops - target_flops))
        
        rb_ppl_at_flops = rb_ppl[rb_idx]
        mb_ppl_at_flops = mb_ppl[mb_idx]
        
        if mb_ppl_at_flops > 0:
            improvement = ((mb_ppl_at_flops - rb_ppl_at_flops) / mb_ppl_at_flops) * 100
        else:
            improvement = 0
        
        flops_label = f'{int(target_flops/1e9)}G'
        equal_flops_comparisons[flops_label] = {
            'target_flops': float(target_flops),
            'resnetbk_actual_flops': float(rb_flops[rb_idx]),
            'mamba_actual_flops': float(mb_flops[mb_idx]),
            'resnetbk_ppl': float(rb_ppl_at_flops),
            'mamba_ppl': float(mb_ppl_at_flops),
            'ppl_improvement_pct': float(improvement),
        }
    
    summary['comparison']['equal_flops'] = equal_flops_comparisons
    
    # Check requirements
    summary['requirements_check'] = {}
    
    # Requirement 8.10: ResNet-BK achieves PPL=30 with 2× fewer FLOPs than Mamba
    ppl_30_comp = equal_ppl_comparisons.get('ppl_30', {})
    if ppl_30_comp:
        flops_ratio = ppl_30_comp.get('flops_ratio', 0)
        summary['requirements_check']['req_8_10_2x_fewer_flops_at_ppl30'] = {
            'met': flops_ratio >= 2.0,
            'value': flops_ratio,
            'threshold': 2.0,
            'description': 'ResNet-BK uses 2× fewer FLOPs than Mamba at PPL≈30'
        }
    
    # Requirement 8.11: At equal FLOPs, ResNet-BK has 30% lower PPL
    flops_5g_comp = equal_flops_comparisons.get('5G', {})
    if flops_5g_comp:
        improvement = flops_5g_comp.get('ppl_improvement_pct', 0)
        summary['requirements_check']['req_8_11_30pct_lower_ppl_at_equal_flops'] = {
            'met': improvement >= 30.0,
            'value': improvement,
            'threshold': 30.0,
            'description': 'ResNet-BK has 30% lower PPL than Mamba at equal FLOPs'
        }
    
    return summary


def print_summary(summary: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("Dynamic Efficiency Summary")
    print("=" * 80)
    
    print("\nResNet-BK Results:")
    print("-" * 80)
    print(f"Best PPL: {summary['resnetbk']['best_ppl']:.2f} "
          f"(at {summary['resnetbk']['best_ppl_flops']/1e9:.2f} GFLOPs)")
    print(f"Lowest FLOPs: {summary['resnetbk']['lowest_flops']/1e9:.2f} GFLOPs "
          f"(PPL {summary['resnetbk']['lowest_flops_ppl']:.2f})")
    print(f"Mean: PPL {summary['resnetbk']['mean_ppl']:.2f}, "
          f"{summary['resnetbk']['mean_flops']/1e9:.2f} GFLOPs")
    print(f"Configurations: {summary['resnetbk']['num_configs']}")
    
    print("\nMamba Results:")
    print("-" * 80)
    print(f"Best PPL: {summary['mamba']['best_ppl']:.2f} "
          f"(at {summary['mamba']['best_ppl_flops']/1e9:.2f} GFLOPs)")
    print(f"Lowest FLOPs: {summary['mamba']['lowest_flops']/1e9:.2f} GFLOPs "
          f"(PPL {summary['mamba']['lowest_flops_ppl']:.2f})")
    print(f"Mean: PPL {summary['mamba']['mean_ppl']:.2f}, "
          f"{summary['mamba']['mean_flops']/1e9:.2f} GFLOPs")
    print(f"Configurations: {summary['mamba']['num_configs']}")
    
    print("\nComparison at Equal PPL:")
    print("-" * 80)
    print(f"{'Target PPL':<12} {'RB FLOPs':<15} {'Mamba FLOPs':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for key, comp in summary['comparison']['equal_ppl'].items():
        print(f"{comp['target_ppl']:<12.1f} "
              f"{comp['resnetbk_flops']/1e9:<15.2f} "
              f"{comp['mamba_flops']/1e9:<15.2f} "
              f"{comp['flops_ratio']:<10.2f}×")
    
    print("\nComparison at Equal FLOPs:")
    print("-" * 80)
    print(f"{'Target FLOPs':<15} {'RB PPL':<12} {'Mamba PPL':<12} {'Improvement':<15}")
    print("-" * 80)
    
    for key, comp in summary['comparison']['equal_flops'].items():
        print(f"{comp['target_flops']/1e9:<15.1f} "
              f"{comp['resnetbk_ppl']:<12.2f} "
              f"{comp['mamba_ppl']:<12.2f} "
              f"{comp['ppl_improvement_pct']:<15.1f}%")
    
    # Print requirements check
    if summary.get('requirements_check'):
        print("\nRequirements Verification:")
        print("-" * 80)
        
        for req_name, req_data in summary['requirements_check'].items():
            status = "✓ PASS" if req_data['met'] else "✗ FAIL"
            print(f"{req_data['description']}")
            print(f"  {status}: Value = {req_data['value']:.2f}, Threshold = {req_data['threshold']:.2f}")
    
    print("=" * 80)



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate dynamic efficiency comparison graph"
    )
    
    # Input
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/efficiency',
        help='Directory containing efficiency results'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Use simulated data for testing'
    )
    parser.add_argument(
        '--num_configs',
        type=int,
        default=10,
        help='Number of configurations to simulate'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='results/efficiency_graph',
        help='Output file path (without extension)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for raster formats'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='pdf',
        choices=['pdf', 'svg', 'png', 'eps'],
        help='Output format'
    )
    
    # Options
    parser.add_argument(
        '--save_summary',
        action='store_true',
        help='Save summary statistics to JSON'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for simulation'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 80)
    print("Dynamic Efficiency Graph Generator")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.simulate:
        print("\nGenerating simulated data...")
        print(f"Number of configurations: {args.num_configs}")
        
        # Simulate data for both models
        resnetbk_results = simulate_efficiency_results(
            'resnetbk',
            args.num_configs,
            args.seed
        )
        
        mamba_results = simulate_efficiency_results(
            'mamba',
            args.num_configs,
            args.seed
        )
        
        print("✓ Simulated data generated")
        
    else:
        print(f"\nLoading results from: {args.results_dir}")
        results_dir = Path(args.results_dir)
        
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            print("Use --simulate flag to generate with simulated data")
            return
        
        # Load results
        resnetbk_dict = load_efficiency_results(results_dir, 'resnetbk')
        mamba_dict = load_efficiency_results(results_dir, 'mamba')
        
        if not resnetbk_dict and not mamba_dict:
            print("Error: No results found in directory")
            print("Use --simulate flag to generate with simulated data")
            return
        
        # Convert to list format
        resnetbk_results = list(resnetbk_dict.values())
        mamba_results = list(mamba_dict.values())
        
        print(f"✓ Loaded ResNet-BK results for {len(resnetbk_results)} configurations")
        print(f"✓ Loaded Mamba results for {len(mamba_results)} configurations")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary = generate_summary_statistics(resnetbk_results, mamba_results)
    print_summary(summary)
    
    # Save summary if requested
    if args.save_summary:
        summary_path = Path(args.output).with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Summary saved: {summary_path}")
    
    # Generate graph
    print(f"\nGenerating efficiency graph...")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print(f"DPI: {args.dpi}")
    
    plot_efficiency_graph(
        resnetbk_results,
        mamba_results,
        args.output,
        dpi=args.dpi,
        format=args.format
    )
    
    print("\n" + "=" * 80)
    print("✓ Efficiency graph generation complete!")
    print("=" * 80)
    
    # Print file locations
    output_path = Path(args.output)
    print(f"\nGenerated files:")
    if args.format == 'pdf':
        print(f"  - {output_path.with_suffix('.pdf')}")
    elif args.format == 'svg':
        print(f"  - {output_path.with_suffix('.svg')}")
    elif args.format == 'eps':
        print(f"  - {output_path.with_suffix('.eps')}")
    print(f"  - {output_path.with_suffix('.png')}")
    
    if args.save_summary:
        print(f"  - {output_path.with_suffix('.json')}")


if __name__ == '__main__':
    main()

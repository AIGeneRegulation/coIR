"""
run_pilot.py — Run the full pilot analysis on SG-NEx A549 direct RNA data.

Steps:
1. Parse introns from GENCODE annotation
2. Detect IR events from aligned BAM
3. Find intron pairs for co-retention testing
4. Run co-retention analysis
5. Generate summary statistics and initial plots

Usage:
    python src/run_pilot.py --bam data/aligned/SGNEx_A549_directRNA_replicate1_run1.bam \
                            --gtf data/annotations/gencode.v44.annotation.gtf \
                            --outdir results/pilot/
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir_utils import (
    parse_introns_from_gtf,
    compute_ir_from_bam,
    ir_events_to_dataframe,
)
from coretention import (
    find_coretention_pairs,
    analyze_coretention,
    coretention_to_dataframe,
    summarize_coretention,
)


def main():
    parser = argparse.ArgumentParser(description="Pilot IR + co-retention analysis")
    parser.add_argument("--bam", required=True, help="Path to aligned BAM file")
    parser.add_argument("--gtf", required=True, help="Path to GENCODE GTF")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--min-ir-ratio", type=float, default=0.05,
                        help="Minimum IR ratio to consider an intron retained (default: 0.05)")
    parser.add_argument("--min-coverage", type=int, default=5,
                        help="Minimum reads spanning an intron (default: 5)")
    parser.add_argument("--min-coret-reads", type=int, default=10,
                        help="Minimum reads spanning both introns for co-retention (default: 10)")
    parser.add_argument("--max-pair-distance", type=int, default=5,
                        help="Maximum intron index distance for pair testing (default: 5)")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)
    
    sample_name = os.path.basename(args.bam).replace('.bam', '').replace('.sorted', '')
    
    # ── Step 1: Parse introns ──
    print("=" * 60)
    print("STEP 1: Parsing introns from GTF")
    print("=" * 60)
    t0 = time.time()
    introns = parse_introns_from_gtf(args.gtf)
    print(f"  Time: {time.time() - t0:.1f}s")
    
    # ── Step 2: IR detection ──
    print("\n" + "=" * 60)
    print("STEP 2: Detecting intron retention from BAM")
    print("=" * 60)
    t0 = time.time()
    ir_events = compute_ir_from_bam(
        args.bam, introns,
        min_coverage=args.min_coverage
    )
    print(f"  Detected {len(ir_events)} IR-measurable introns")
    print(f"  Time: {time.time() - t0:.1f}s")
    
    # Convert to DataFrame and save
    ir_df = ir_events_to_dataframe(ir_events)
    ir_path = os.path.join(args.outdir, f"{sample_name}_ir_events.tsv")
    ir_df.to_csv(ir_path, sep='\t', index=False)
    print(f"  Saved to {ir_path}")
    
    # Summary stats
    retained = ir_df[ir_df['ir_ratio'] >= args.min_ir_ratio]
    print(f"\n  IR summary (ratio >= {args.min_ir_ratio}):")
    print(f"    Total introns measurable: {len(ir_df)}")
    print(f"    Introns with IR: {len(retained)}")
    print(f"    Genes with IR: {retained['gene_id'].nunique()}")
    print(f"    Median IR ratio (retained): {retained['ir_ratio'].median():.3f}")
    
    # ── Step 3: Find co-retention pairs ──
    print("\n" + "=" * 60)
    print("STEP 3: Identifying intron pairs for co-retention analysis")
    print("=" * 60)
    
    # Only test pairs where at least one intron shows retention
    retained_introns = []
    retained_genes = set(retained['gene_id'].values)
    for intron in introns:
        if intron.gene_id in retained_genes:
            retained_introns.append(intron)
    
    pairs = find_coretention_pairs(
        retained_introns,
        max_pair_distance=args.max_pair_distance
    )
    print(f"  Found {len(pairs)} intron pairs to test across {len(retained_genes)} genes")
    
    # ── Step 4: Co-retention analysis ──
    print("\n" + "=" * 60)
    print("STEP 4: Analyzing co-retention")
    print("=" * 60)
    t0 = time.time()
    coret_results = analyze_coretention(
        args.bam, pairs,
        min_reads=args.min_coret_reads
    )
    print(f"  Time: {time.time() - t0:.1f}s")
    
    coret_df = coretention_to_dataframe(coret_results)
    coret_path = os.path.join(args.outdir, f"{sample_name}_coretention.tsv")
    coret_df.to_csv(coret_path, sep='\t', index=False)
    print(f"  Saved to {coret_path}")
    
    # ── Step 5: Summary and plots ──
    print("\n" + "=" * 60)
    print("STEP 5: Summary statistics")
    print("=" * 60)
    
    if len(coret_df) > 0:
        summary = summarize_coretention(coret_df, fdr_threshold=0.05)
        summary_path = os.path.join(args.outdir, f"{sample_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"  Pairs tested: {summary['total_pairs_tested']}")
        print(f"  Significant pairs (FDR<0.05): {summary['significant_pairs']}")
        print(f"  Positive co-retention: {summary['positive_coretention']}")
        print(f"  Negative co-retention: {summary['negative_coretention']}")
        print(f"  Genes with co-retention: {summary['genes_with_coretention']}")
        if 'adjacent_enrichment_pvalue' in summary:
            print(f"  Adjacent pair enrichment p-value: {summary['adjacent_enrichment_pvalue']:.2e}")
        
        # Generate plots
        _plot_ir_distribution(ir_df, args.outdir, sample_name, args.min_ir_ratio)
        _plot_coretention(coret_df, args.outdir, sample_name)
    else:
        print("  No intron pairs passed filters — try lowering --min-coret-reads")
        _plot_ir_distribution(ir_df, args.outdir, sample_name, args.min_ir_ratio)
        # Write zero-result summary so downstream scripts don't skip this sample
        summary = {
            'total_pairs_tested': 0, 'significant_pairs': 0,
            'positive_coretention': 0, 'negative_coretention': 0,
            'genes_with_coretention': 0, 'adjacent_enrichment_pvalue': 1.0,
        }
        summary_path = os.path.join(args.outdir, f"{sample_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PILOT ANALYSIS COMPLETE")
    print(f"Results in: {args.outdir}")
    print("=" * 60)


def _plot_ir_distribution(ir_df, outdir, sample_name, min_ratio):
    """Plot IR ratio distribution (cf. IRFinder paper Figure 1a)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: histogram of IR ratios
    ax = axes[0]
    retained = ir_df[ir_df['ir_ratio'] >= min_ratio]
    ax.hist(retained['ir_ratio'], bins=50, color='steelblue', edgecolor='white')
    ax.set_xlabel('IR ratio')
    ax.set_ylabel('Number of introns')
    ax.set_title(f'{sample_name}\n{len(retained)} retained introns (ratio ≥ {min_ratio})')
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    ax.legend()
    
    # Panel B: number of retained introns per gene
    ax = axes[1]
    gene_counts = retained.groupby('gene_name').size()
    if len(gene_counts) > 0 and not np.isnan(gene_counts.max()):
        ax.hist(gene_counts, bins=range(1, min(int(gene_counts.max()) + 2, 30)),
                color='coral', edgecolor='white')
    ax.set_xlabel('Retained introns per gene')
    ax.set_ylabel('Number of genes')
    ax.set_title(f'Genes with multiple retained introns\n({(gene_counts >= 2).sum()} genes with ≥2)')
    
    plt.tight_layout()
    fig_path = os.path.join(outdir, "figures", f"{sample_name}_ir_distribution.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved IR distribution plot to {fig_path}")


def _plot_coretention(coret_df, outdir, sample_name):
    """Plot co-retention analysis results."""
    if len(coret_df) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: phi coefficient distribution
    ax = axes[0]
    ax.hist(coret_df['phi_coefficient'], bins=50, color='steelblue', edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    sig = coret_df[coret_df['fdr'] < 0.05]
    if len(sig) > 0:
        ax.hist(sig['phi_coefficient'], bins=50, color='coral', edgecolor='white',
                alpha=0.7, label=f'FDR<0.05 (n={len(sig)})')
        ax.legend()
    ax.set_xlabel('Phi coefficient')
    ax.set_ylabel('Number of intron pairs')
    ax.set_title('Co-retention correlation')
    
    # Panel B: phi coefficient vs genomic distance between introns
    ax = axes[1]
    distance = coret_df['intron_b_index'] - coret_df['intron_a_index']
    ax.scatter(distance, coret_df['phi_coefficient'],
               alpha=0.3, s=10, c='steelblue')
    if len(sig) > 0:
        sig_dist = sig['intron_b_index'] - sig['intron_a_index']
        ax.scatter(sig_dist, sig['phi_coefficient'],
                   alpha=0.6, s=15, c='coral', label='FDR<0.05')
        ax.legend()
    ax.set_xlabel('Intron index distance')
    ax.set_ylabel('Phi coefficient')
    ax.set_title('Co-retention vs intron distance')
    ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
    
    # Panel C: adjacent vs non-adjacent comparison
    ax = axes[2]
    adj_phi = coret_df[coret_df['adjacent']]['phi_coefficient']
    nonadj_phi = coret_df[~coret_df['adjacent']]['phi_coefficient']
    
    data_to_plot = []
    labels = []
    if len(adj_phi) > 0:
        data_to_plot.append(adj_phi)
        labels.append(f'Adjacent\n(n={len(adj_phi)})')
    if len(nonadj_phi) > 0:
        data_to_plot.append(nonadj_phi)
        labels.append(f'Non-adjacent\n(n={len(nonadj_phi)})')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['coral', 'steelblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Phi coefficient')
        ax.set_title('Adjacent vs non-adjacent\nintron co-retention')
        ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(outdir, "figures", f"{sample_name}_coretention.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved co-retention plot to {fig_path}")


if __name__ == "__main__":
    main()

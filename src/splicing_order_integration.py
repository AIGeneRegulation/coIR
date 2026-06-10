"""
splicing_order_integration.py — Integrate co-retention results with Kim et al. 2023 splicing order.

Kim et al. 2023 (GSE232455) profiled co-transcriptional splicing order in K562 cells using
direct RNA nanopore of chromatin-associated RNA. For each gene, they determined the typical
order in which introns are spliced.

For each significant co-retained intron pair, this script looks up whether the two introns
are consecutively ordered in Kim et al.'s splicing order, and tests whether co-retained pairs
are enriched for consecutive splicing positions vs. all tested pairs.

Hypothesis: if co-retained introns are functionally linked, they should tend to occupy
adjacent positions in the normal splicing order.

Usage:
    python src/splicing_order_integration.py \
        --coretention results/SGNex_K562_directRNA_replicate1_run1/..._coretention.tsv \
        --kim-dir data/kim_splicing_order/ \
        --outdir results/SGNex_K562_directRNA_replicate1_run1/ \
        [--fdr-threshold 0.05]
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_kim_splicing_order(kim_dir: str) -> pd.DataFrame | None:
    """
    Load Kim et al. processed splicing order data.

    Priority:
      1. Preprocessed splicing_order_table.tsv (from src/preprocess_kim_data.py)
         Columns: gene_name, intron_index, p_spliced, mean_order_position, n_reads
      2. Any TSV/CSV with recognisable order/splicing columns (legacy fallback)

    Returns a DataFrame with at least: gene_name, intron_index, mean_order_position
    Or None if no suitable file is found.
    """
    # 1. Preprocessed table
    preprocessed = os.path.join(kim_dir, "splicing_order_table.tsv")
    if os.path.exists(preprocessed):
        df = pd.read_csv(preprocessed, sep='\t')
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        print(f"  Loaded preprocessed: splicing_order_table.tsv — {len(df)} rows")
        return df

    # 2. Legacy fallback: scan for TSV/CSV with order-like columns
    candidates = (
        glob.glob(os.path.join(kim_dir, "*.tsv")) +
        glob.glob(os.path.join(kim_dir, "*.txt")) +
        glob.glob(os.path.join(kim_dir, "*.csv")) +
        glob.glob(os.path.join(kim_dir, "code", "*.tsv")) +
        glob.glob(os.path.join(kim_dir, "code", "results", "*.tsv"))
    )

    order_keywords = ['order', 'splicing', 'intron', 'rank']

    for fpath in sorted(candidates):
        fname = os.path.basename(fpath).lower()
        if not any(kw in fname for kw in order_keywords):
            continue
        if any(skip in fname for skip in ('readme', 'log', 'index')):
            continue
        try:
            for sep in ('\t', ','):
                try:
                    df = pd.read_csv(fpath, sep=sep, nrows=5)
                    if len(df.columns) >= 2:
                        break
                except Exception:
                    continue
            df = pd.read_csv(fpath, sep=sep, low_memory=False)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            print(f"  Loaded: {os.path.basename(fpath)} — columns: {list(df.columns)[:8]}")
            return df
        except Exception as e:
            print(f"  Could not load {fpath}: {e}")

    print("  No preprocessed splicing_order_table.tsv found.")
    print("  Run preprocess_kim_data.py first to generate splicing_order_table.tsv.")
    return None


def normalise_gene_name(s: str) -> str:
    return str(s).upper().strip()


def build_order_lookup(kim_df: pd.DataFrame) -> dict:
    """
    Build a dict: {(gene_name, intron_index) -> mean_order_position}
    Tries to infer which columns contain gene name, intron index, and order.
    """
    cols = set(kim_df.columns)

    # Gene column
    gene_col = next((c for c in ['gene_name', 'gene', 'symbol', 'genename', 'name'] if c in cols), None)
    if gene_col is None:
        gene_col = kim_df.columns[0]

    # Intron index column
    idx_col = next((c for c in ['intron_index', 'intron_number', 'intron_rank',
                                  'exon_number', 'intron_id', 'index'] if c in cols), None)
    if idx_col is None:
        # Try numeric columns
        num_cols = [c for c in kim_df.columns if kim_df[c].dtype in (int, float) and c != gene_col]
        idx_col = num_cols[0] if num_cols else kim_df.columns[1]

    # Order position column
    order_col = next((c for c in ['mean_order_position', 'order_position', 'mean_order',
                                    'order', 'rank', 'splicing_order', 'position', 'mean_rank']
                      if c in cols), None)
    if order_col is None:
        num_cols = [c for c in kim_df.columns if kim_df[c].dtype == float and c not in (gene_col, idx_col)]
        order_col = num_cols[0] if num_cols else (idx_col if idx_col != kim_df.columns[0] else kim_df.columns[-1])

    print(f"  Using columns: gene={gene_col}, intron_idx={idx_col}, order={order_col}")

    lookup = {}
    for _, row in kim_df.iterrows():
        gene = normalise_gene_name(row[gene_col])
        try:
            idx = int(row[idx_col])
            order = float(row[order_col])
            lookup[(gene, idx)] = order
        except (ValueError, TypeError):
            continue
    return lookup


def are_consecutive_in_order(order_a: float, order_b: float, max_gap: float = 1.5) -> bool:
    """True if the two order positions are within max_gap of each other."""
    return abs(order_a - order_b) <= max_gap


def enrich_consecutive(coret_df: pd.DataFrame, order_lookup: dict,
                        fdr_threshold: float = 0.05) -> dict:
    """
    For significant vs. non-significant co-retention pairs, test enrichment for
    consecutive splicing order positions.

    2x2 table:
                       consecutive    not consecutive
    co-retained sig  |      a       |       b       |
    not sig          |      c       |       d       |
    """
    sig = coret_df['fdr'] < fdr_threshold

    def check_pair(row):
        gene = normalise_gene_name(row['gene_name'])
        oa = order_lookup.get((gene, int(row['intron_a_index'])))
        ob = order_lookup.get((gene, int(row['intron_b_index'])))
        if oa is None or ob is None:
            return None
        return are_consecutive_in_order(oa, ob)

    coret_df = coret_df.copy()
    coret_df['_consec'] = coret_df.apply(check_pair, axis=1)

    testable = coret_df.dropna(subset=['_consec']).copy()
    # Force bool dtype: object-column booleans give wrong results with ~ operator
    # (~True = -2 in Python, which is truthy, causing all rows to be "not consecutive")
    testable['_consec'] = testable['_consec'].astype(bool)
    n_lookup = len(testable)
    print(f"  Pairs with splicing order data: {n_lookup} / {len(coret_df)}")
    if n_lookup == 0:
        return {'error': 'no pairs found in Kim et al. data'}

    sig_consec = int(((testable['fdr'] < fdr_threshold) &  testable['_consec']).sum())
    sig_not    = int(((testable['fdr'] < fdr_threshold) & ~testable['_consec']).sum())
    ns_consec  = int(((testable['fdr'] >= fdr_threshold) &  testable['_consec']).sum())
    ns_not     = int(((testable['fdr'] >= fdr_threshold) & ~testable['_consec']).sum())

    table = [[sig_consec, sig_not], [ns_consec, ns_not]]
    odds_ratio, pvalue = stats.fisher_exact(table, alternative='greater')

    return {
        'n_pairs_with_order_data': n_lookup,
        'sig_consecutive': sig_consec,
        'sig_not_consecutive': sig_not,
        'ns_consecutive': ns_consec,
        'ns_not_consecutive': ns_not,
        'pct_sig_consecutive': sig_consec / max(sig_consec + sig_not, 1) * 100,
        'pct_ns_consecutive': ns_consec / max(ns_consec + ns_not, 1) * 100,
        'odds_ratio': odds_ratio,
        'pvalue': pvalue,
        'table': testable,
    }


def main():
    parser = argparse.ArgumentParser(description="Integrate co-retention with Kim 2023 splicing order")
    parser.add_argument("--coretention", required=True)
    parser.add_argument("--kim-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sample_name = os.path.basename(args.coretention).replace('_coretention.tsv', '')

    print("Loading co-retention results...")
    coret_df = pd.read_csv(args.coretention, sep='\t')
    print(f"  {len(coret_df)} pairs, {(coret_df['fdr'] < args.fdr_threshold).sum()} significant")

    print("\nLoading Kim et al. 2023 splicing order data...")
    kim_df = load_kim_splicing_order(args.kim_dir)
    if kim_df is None:
        print("  No splicing order data found. Download Kim et al. 2023 data and point --kim-dir to it.")
        print("  If data is in a non-standard format, check the expected columns in load_kim_splicing_order().")
        return

    print(f"  {len(kim_df)} entries")
    order_lookup = build_order_lookup(kim_df)
    print(f"  {len(order_lookup)} (gene, intron_index) entries in lookup")

    print("\nTesting consecutive splicing order enrichment...")
    result = enrich_consecutive(coret_df, order_lookup, args.fdr_threshold)

    if 'error' in result:
        print(f"  {result['error']}")
        return

    print(f"  Significant pairs that are consecutive in splicing order: "
          f"{result['sig_consecutive']} / {result['sig_consecutive'] + result['sig_not_consecutive']} "
          f"({result['pct_sig_consecutive']:.1f}%)")
    print(f"  Non-significant pairs consecutive:  "
          f"{result['ns_consecutive']} / {result['ns_consecutive'] + result['ns_not_consecutive']} "
          f"({result['pct_ns_consecutive']:.1f}%)")
    print(f"  Odds ratio: {result['odds_ratio']:.2f}  p={result['pvalue']:.3e}")

    # Save annotated pairs
    out_df = result['table'].drop(columns=['_consec'], errors='ignore')
    out_tsv = os.path.join(args.outdir, f'{sample_name}_splicing_order_enrichment.tsv')
    out_df.to_csv(out_tsv, sep='\t', index=False)
    print(f"\nSaved: {out_tsv}")

    # Summary stats
    stats_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'table'}])
    stats_tsv = os.path.join(args.outdir, f'{sample_name}_splicing_order_stats.tsv')
    stats_df.to_csv(stats_tsv, sep='\t', index=False)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    groups = ['Co-retained\n(FDR<0.05)', 'Not significant']
    pct_consec = [result['pct_sig_consecutive'], result['pct_ns_consecutive']]
    colors = ['coral', 'steelblue']
    bars = ax.bar(groups, pct_consec, color=colors, alpha=0.8)
    ax.set_ylabel('% pairs with consecutive splicing order')
    ax.set_title(f'Splicing order enrichment\nOR={result["odds_ratio"]:.2f}  p={result["pvalue"]:.2e}')
    for bar, pct in zip(bars, pct_consec):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', fontsize=10)

    ax = axes[1]
    table_data = result['table']
    sig_mask = table_data['fdr'] < args.fdr_threshold
    sig_consec_mask = sig_mask & (table_data.get('_consec', False) == True)
    ax.scatter(table_data['intron_a_index'], table_data['intron_b_index'],
               c='lightgrey', s=10, alpha=0.5, label='All pairs')
    if sig_mask.any():
        ax.scatter(table_data[sig_mask]['intron_a_index'],
                   table_data[sig_mask]['intron_b_index'],
                   c='coral', s=20, alpha=0.7, label=f'Sig (n={sig_mask.sum()})')
    ax.set_xlabel('Intron A index')
    ax.set_ylabel('Intron B index')
    ax.set_title('Co-retention pairs by intron position')
    ax.legend(fontsize=8)

    plt.suptitle(f'{sample_name}\nKim et al. 2023 splicing order integration', fontsize=10)
    plt.tight_layout()
    out_fig = os.path.join(args.outdir, 'figures', f'{sample_name}_splicing_order.png')
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()

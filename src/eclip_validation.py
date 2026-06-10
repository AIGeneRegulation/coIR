"""
eclip_validation.py — Validate RBP binding at co-retained introns using ENCODE eCLIP data.

For each RBP with eCLIP data, checks whether peaks fall within 200bp of intron splice sites
or within the intron body, then compares binding frequency across three intron classes:
  (a) co-retained  — in a significant co-retention pair (FDR < threshold)
  (b) indep_retained — IR ratio >= ir_threshold but not in any significant pair
  (c) non_retained  — IR ratio < ir_threshold

Fisher's exact test: co-retained vs independently-retained for each RBP.

Usage:
    python src/eclip_validation.py \
        --coretention results/SGNex_K562_directRNA_replicate1_run1/..._coretention.tsv \
        --eclip-dir data/encode_eclip/K562/ \
        --cell-line K562 \
        --outdir results/mechanistic/eclip/ \
        [--fdr-threshold 0.05] [--ir-threshold 0.05] [--window 200]
"""

import argparse
import glob
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


WINDOW = 200  # bp around splice sites


def parse_coord(intron_str: str):
    """Parse 'chr17:64500882-64500986' → (chrom, start, end)."""
    chrom, rest = intron_str.rsplit(':', 1)
    start, end = rest.split('-')
    return chrom, int(start), int(end)


def load_eclip_peaks(bed_path: str) -> dict:
    """
    Load eCLIP narrowPeak BED. Returns {chrom: (sorted_starts, sorted_ends)}.
    Uses parallel sorted arrays for fast bisect-based overlap queries.
    """
    raw: dict = defaultdict(list)
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            try:
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            except ValueError:
                continue
            raw[chrom].append((start, end))

    index = {}
    for chrom, peaks in raw.items():
        peaks.sort()
        index[chrom] = (
            [p[0] for p in peaks],  # starts
            [p[1] for p in peaks],  # ends
        )
    return index


def has_peak_in_window(index: dict, chrom: str, qstart: int, qend: int) -> bool:
    """True if any eCLIP peak overlaps [qstart, qend)."""
    if chrom not in index:
        return False
    starts, ends = index[chrom]
    # Peaks that start before qend and end after qstart
    # Find rightmost start < qend
    hi = bisect_left(starts, qend)
    if hi == 0:
        return False
    # Check from hi-1 downward: if ends[i] > qstart it overlaps
    # Only need to check back until starts[i] < qstart (no earlier peak can overlap qend)
    lo = bisect_left(starts, qstart - 1)  # rough lower bound
    for i in range(hi - 1, max(lo - 1, -1), -1):
        if ends[i] > qstart:
            return True
    return False


def intron_is_bound(index: dict, chrom: str, istart: int, iend: int, window: int) -> bool:
    """True if any eCLIP peak overlaps the 5'SS ±window, 3'SS ±window, or intron body."""
    # 5' splice site
    if has_peak_in_window(index, chrom, istart - window, istart + window):
        return True
    # 3' splice site
    if has_peak_in_window(index, chrom, iend - window, iend + window):
        return True
    # intron body
    if has_peak_in_window(index, chrom, istart, iend):
        return True
    return False


def classify_introns(coret_df: pd.DataFrame,
                     fdr_threshold: float, ir_threshold: float) -> pd.DataFrame:
    """
    Return a DataFrame with one row per unique intron (from intron_a and intron_b columns):
      chrom, start, end, ir_ratio, intron_class (co_retained / indep_retained / non_retained)
    """
    sig_mask = coret_df['fdr'] < fdr_threshold
    sig_df   = coret_df[sig_mask]

    # Introns in any significant pair
    co_retained_coords: set = set()
    for col in ('intron_a', 'intron_b'):
        co_retained_coords.update(sig_df[col].values)

    # Build per-intron IR ratio lookup (first occurrence wins)
    ir_lookup: dict = {}
    for _, row in coret_df.iterrows():
        if row['intron_a'] not in ir_lookup:
            ir_lookup[row['intron_a']] = float(row['ir_ratio_a'])
        if row['intron_b'] not in ir_lookup:
            ir_lookup[row['intron_b']] = float(row['ir_ratio_b'])

    rows = []
    for coord, ir_ratio in ir_lookup.items():
        try:
            chrom, start, end = parse_coord(coord)
        except Exception:
            continue
        if coord in co_retained_coords:
            cls = 'co_retained'
        elif ir_ratio >= ir_threshold:
            cls = 'indep_retained'
        else:
            cls = 'non_retained'
        rows.append({'coord': coord, 'chrom': chrom, 'start': start, 'end': end,
                     'ir_ratio': ir_ratio, 'intron_class': cls})
    return pd.DataFrame(rows)


def run_one_rbp(rbp: str, bed_path: str, intron_df: pd.DataFrame,
                window: int) -> dict:
    """Run eCLIP binding analysis for one RBP. Returns stats dict."""
    print(f"  Loading eCLIP peaks for {rbp}...", flush=True)
    index = load_eclip_peaks(bed_path)
    n_peaks = sum(len(v[0]) for v in index.values())
    print(f"    {n_peaks} peaks across {len(index)} chromosomes", flush=True)

    bound = intron_df.apply(
        lambda row: intron_is_bound(index, row['chrom'], row['start'], row['end'], window),
        axis=1
    )
    intron_df = intron_df.copy()
    intron_df['bound'] = bound

    result = {'rbp': rbp, 'n_peaks': n_peaks}
    class_stats = {}
    for cls in ('co_retained', 'indep_retained', 'non_retained'):
        sub = intron_df[intron_df['intron_class'] == cls]
        n_total = len(sub)
        n_bound = int(sub['bound'].sum())
        pct = n_bound / n_total * 100 if n_total > 0 else 0.0
        class_stats[cls] = {'n_total': n_total, 'n_bound': n_bound, 'pct_bound': pct}
        result[f'{cls}_n_total'] = n_total
        result[f'{cls}_n_bound'] = n_bound
        result[f'{cls}_pct_bound'] = pct

    # Fisher's exact: co_retained vs indep_retained
    co  = class_stats['co_retained']
    ir  = class_stats['indep_retained']
    if co['n_total'] > 0 and ir['n_total'] > 0:
        table = [
            [co['n_bound'],     co['n_total']  - co['n_bound']],
            [ir['n_bound'],     ir['n_total']  - ir['n_bound']],
        ]
        or_, pval = stats.fisher_exact(table, alternative='greater')
    else:
        or_, pval = float('nan'), float('nan')
    result['fisher_or']    = or_
    result['fisher_pvalue'] = pval

    print(f"    co_retained: {co['n_bound']}/{co['n_total']} ({co['pct_bound']:.1f}%)  "
          f"indep: {ir['n_bound']}/{ir['n_total']} ({ir['pct_bound']:.1f}%)  "
          f"OR={or_:.2f}  p={pval:.2e}", flush=True)
    return result


def fdr_correct(pvals: list) -> list:
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    fdr = [0.0] * n
    cummin = 1.0
    for rank, idx in enumerate(reversed(order)):
        adj = pvals[idx] * n / (n - rank)
        cummin = min(cummin, adj, 1.0)
        fdr[idx] = cummin
    return fdr


def plot_results(results_df: pd.DataFrame, outdir: str, sample_name: str):
    rbps = results_df['rbp'].tolist()
    n = len(rbps)
    if n == 0:
        return

    x = np.arange(n)
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 1.4), 5))

    # Panel A: % bound by class
    ax = axes[0]
    pct_co  = results_df['co_retained_pct_bound'].fillna(0).tolist()
    pct_ir  = results_df['indep_retained_pct_bound'].fillna(0).tolist()
    pct_nr  = results_df['non_retained_pct_bound'].fillna(0).tolist()
    ax.bar(x - width, pct_co, width, label='Co-retained', color='coral', alpha=0.85)
    ax.bar(x,         pct_ir, width, label='Indep. retained', color='steelblue', alpha=0.85)
    ax.bar(x + width, pct_nr, width, label='Non-retained', color='lightgrey', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(rbps, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('% introns with eCLIP peak (±200bp SS or body)')
    ax.set_title('eCLIP binding frequency by intron class')
    ax.legend(fontsize=8)

    # Panel B: volcano-style OR vs –log10(FDR)
    ax = axes[1]
    valid = results_df.dropna(subset=['fisher_or', 'fdr'])
    if len(valid) > 0:
        log_or  = np.log2(valid['fisher_or'].clip(lower=0.001))
        neg_log = -np.log10(valid['fdr'].clip(lower=1e-10))
        colors  = ['coral' if p < 0.05 else 'steelblue' for p in valid['fdr']]
        ax.scatter(log_or, neg_log, c=colors, s=60, alpha=0.8)
        ax.axhline(-np.log10(0.05), color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
        for _, row in valid.iterrows():
            ax.annotate(row['rbp'],
                        (np.log2(max(row['fisher_or'], 0.001)),
                         -np.log10(max(row['fdr'], 1e-10))),
                        fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('log2(OR)  co-retained vs indep-retained')
    ax.set_ylabel('–log10(FDR)')
    ax.set_title('Enrichment: co-retained vs independently retained')

    plt.suptitle(f'{sample_name} — eCLIP binding enrichment', fontsize=10)
    plt.tight_layout()
    fig_dir = os.path.join(outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f'{sample_name}_eclip_enrichment.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coretention', required=True,
                        help='Co-retention TSV (*_coretention.tsv)')
    parser.add_argument('--eclip-dir', required=True,
                        help='Dir with <RBP>.bed files for the target cell line')
    parser.add_argument('--cell-line', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--fdr-threshold', type=float, default=0.05)
    parser.add_argument('--ir-threshold', type=float, default=0.05,
                        help='Min IR ratio for independently retained class')
    parser.add_argument('--window', type=int, default=WINDOW,
                        help='bp around splice sites to check for eCLIP peaks')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sample_name = os.path.basename(args.coretention).replace('_coretention.tsv', '')

    print(f"Loading co-retention results: {args.coretention}")
    coret_df = pd.read_csv(args.coretention, sep='\t')
    print(f"  {len(coret_df)} pairs, "
          f"{(coret_df['fdr'] < args.fdr_threshold).sum()} significant")

    print("\nClassifying introns...")
    intron_df = classify_introns(coret_df, args.fdr_threshold, args.ir_threshold)
    counts = intron_df['intron_class'].value_counts()
    print(f"  co_retained:   {counts.get('co_retained', 0)}")
    print(f"  indep_retained:{counts.get('indep_retained', 0)}")
    print(f"  non_retained:  {counts.get('non_retained', 0)}")

    # Find eCLIP BED files
    bed_files = sorted(glob.glob(os.path.join(args.eclip_dir, '*.bed')))
    if not bed_files:
        print(f"No BED files found in {args.eclip_dir}")
        return
    print(f"\nFound {len(bed_files)} eCLIP BED file(s):")
    for b in bed_files:
        print(f"  {os.path.basename(b)}")

    results = []
    for bed_path in bed_files:
        fname = os.path.basename(bed_path)
        # Infer RBP from canonical filename (<RBP>.bed or <RBP>_<accession>.bed)
        rbp = fname.split('_')[0].replace('.bed', '')
        print(f"\n--- {rbp} ---")
        try:
            res = run_one_rbp(rbp, bed_path, intron_df, args.window)
            results.append(res)
        except Exception as e:
            print(f"  ERROR: {e}")

    if not results:
        print("No results produced.")
        return

    results_df = pd.DataFrame(results)
    # FDR-correct across RBPs
    pvals = results_df['fisher_pvalue'].fillna(1.0).tolist()
    results_df['fdr'] = fdr_correct(pvals)

    out_tsv = os.path.join(args.outdir, f'{sample_name}_eclip_enrichment.tsv')
    results_df.to_csv(out_tsv, sep='\t', index=False)
    print(f"\nSaved: {out_tsv}")

    print("\nTop hits (FDR < 0.05):")
    sig = results_df[results_df['fdr'] < 0.05].sort_values('fisher_pvalue')
    if len(sig):
        cols = ['rbp', 'co_retained_pct_bound', 'indep_retained_pct_bound',
                'fisher_or', 'fisher_pvalue', 'fdr']
        print(sig[cols].to_string(index=False))
    else:
        print("  None (FDR < 0.05)")

    plot_results(results_df, args.outdir, sample_name)


if __name__ == '__main__':
    main()

"""
splice_site_strength.py — Compare splice site sequence features across intron retention classes.

Computes per intron:
  1. 5'SS PWM score: 9mer (3bp exonic + 6bp intronic) scored against a log-odds
     position weight matrix derived from known 5'SS frequencies (Shapiro-Senapathy
     consensus MAGGTRAGT — see SS5_PWM below).
  2. U-richness: T fraction in first 30bp of the intron (5'SS-proximal intronic seq).
  3. Polypyrimidine tract (PPT): C+T fraction in last 30bp of the intron (3'SS-proximal).
  4. TIA1 eCLIP signal: max signal value of any TIA1 peak within 200bp of either splice
     site, or 0 if no peak overlaps (only if TIA1.bed is present in --eclip-dir).

Introns are classified into three groups:
  co_retained   — appears in a significant co-retention pair (FDR < --fdr-threshold)
  indep_retained — IR ratio >= --ir-threshold but not in any significant pair
  non_retained  — IR ratio < --ir-threshold

Mann-Whitney U tests (with Benjamini-Hochberg FDR) compare all pairwise group combos
for each metric.

Usage:
    python src/splice_site_strength.py \\
        --coretention results/SGNex_K562_directRNA_replicate4_run1/..._coretention.tsv \\
        --ir-events   results/SGNex_K562_directRNA_replicate4_run1/..._ir_events.tsv \\
        --genome      data/annotations/GRCh38.primary_assembly.genome.fa \\
        --eclip-dir   data/encode_eclip/K562/ \\
        --outdir      results/mechanistic/splice_site_strength/ \\
        [--fdr-threshold 0.05] [--ir-threshold 0.05]
"""

import argparse
import os
from bisect import bisect_left
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfaidx
from scipy import stats

# ── 5'SS position weight matrix ───────────────────────────────────────────────
# 9-mer: positions −3,−2,−1,+1,+2,+3,+4,+5,+6 relative to the GT
# Frequencies derived from Shapiro-Senapathy / Yeo & Burge 2004.
# Scores = log2(freq / 0.25) clipped at −4.64 (pseudocount handles p=0.01).
_SS5_FREQ = {
    # pos:   -3    -2    -1    +1    +2    +3    +4    +5    +6
    'A': [0.33, 0.61, 0.09, 0.01, 0.01, 0.78, 0.57, 0.14, 0.11],
    'C': [0.37, 0.13, 0.04, 0.01, 0.01, 0.07, 0.12, 0.09, 0.09],
    'G': [0.17, 0.13, 0.80, 0.97, 0.01, 0.09, 0.16, 0.68, 0.10],
    'T': [0.13, 0.13, 0.07, 0.01, 0.97, 0.06, 0.15, 0.09, 0.70],
}
# Pseudocount so log2 never hits -inf; clip at floor = log2(0.005/0.25) ≈ -5.64
_PWM_FLOOR = np.log2(0.005 / 0.25)
SS5_PWM: dict[str, list[float]] = {
    nt: [max(np.log2(max(f, 0.005) / 0.25), _PWM_FLOOR) for f in freqs]
    for nt, freqs in _SS5_FREQ.items()
}

_COMP = str.maketrans('ACGTacgt', 'TGCAtgca')


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def score_5ss(seq9: str) -> float:
    """PWM log-odds score for a 9mer 5'SS (3bp exonic + 6bp intronic)."""
    if len(seq9) != 9:
        return float('nan')
    return sum(SS5_PWM.get(nt.upper(), [_PWM_FLOOR] * 9)[i]
               for i, nt in enumerate(seq9))


def _safe_fetch(fasta: pyfaidx.Fasta, chrom: str, start: int, end: int) -> str | None:
    """Fetch sequence [start, end) with bounds checking. Returns None on failure."""
    try:
        rec = fasta[chrom]
        start = max(0, start)
        end = min(len(rec), end)
        if start >= end:
            return None
        return str(rec[start:end]).upper()
    except Exception:
        return None


def extract_5ss_seq(fasta: pyfaidx.Fasta,
                    chrom: str, intron_start: int, intron_end: int,
                    strand: str) -> str | None:
    """
    Extract 9mer for 5'SS (3bp exonic + 6bp intronic).
    + strand: genome[intron_start-3 : intron_start+6]
    - strand: revcomp(genome[intron_end-6 : intron_end+3])
    """
    if strand == '+':
        seq = _safe_fetch(fasta, chrom, intron_start - 3, intron_start + 6)
        return seq
    else:
        seq = _safe_fetch(fasta, chrom, intron_end - 6, intron_end + 3)
        return revcomp(seq) if seq else None


def compute_u_richness(fasta: pyfaidx.Fasta,
                       chrom: str, intron_start: int, intron_end: int,
                       strand: str, window: int = 30) -> float | None:
    """T fraction (proxy for U in RNA) in first 30bp of the intron (5'SS-proximal)."""
    if strand == '+':
        seq = _safe_fetch(fasta, chrom, intron_start, intron_start + window)
    else:
        seq = _safe_fetch(fasta, chrom, intron_end - window, intron_end)
        if seq:
            seq = revcomp(seq)
    if not seq or len(seq) < 5:
        return None
    return seq.count('T') / len(seq)


def compute_ppt_score(fasta: pyfaidx.Fasta,
                      chrom: str, intron_start: int, intron_end: int,
                      strand: str, window: int = 30) -> float | None:
    """C+T fraction in last 30bp of the intron (3'SS-proximal polypyrimidine tract)."""
    if strand == '+':
        seq = _safe_fetch(fasta, chrom, intron_end - window, intron_end)
    else:
        seq = _safe_fetch(fasta, chrom, intron_start, intron_start + window)
        if seq:
            seq = revcomp(seq)
    if not seq or len(seq) < 5:
        return None
    return (seq.count('C') + seq.count('T')) / len(seq)


# ── eCLIP signal index ─────────────────────────────────────────────────────────

def load_eclip_signal(bed_path: str) -> dict:
    """
    Load narrowPeak-format BED (col 7 = signalValue).
    Returns {chrom: (sorted_starts, ends, signals)}.
    """
    raw: dict = defaultdict(list)
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            try:
                chrom = parts[0]
                start, end = int(parts[1]), int(parts[2])
                signal = float(parts[6]) if len(parts) > 6 else 1.0
            except ValueError:
                continue
            raw[chrom].append((start, end, signal))

    index = {}
    for chrom, peaks in raw.items():
        peaks.sort(key=lambda x: x[0])
        index[chrom] = (
            [p[0] for p in peaks],
            [p[1] for p in peaks],
            [p[2] for p in peaks],
        )
    return index


def get_max_eclip_signal(index: dict, chrom: str,
                         intron_start: int, intron_end: int,
                         window: int = 200) -> float:
    """Max signal value of any peak overlapping [ss-window, ss+window] at either SS."""
    if chrom not in index:
        return 0.0
    starts, ends, signals = index[chrom]
    best = 0.0

    for q_start, q_end in [
        (intron_start - window, intron_start + window),
        (intron_end - window,   intron_end + window),
    ]:
        hi = bisect_left(starts, q_end)
        for i in range(hi - 1, -1, -1):
            if starts[i] < q_start - window:
                break
            if ends[i] > q_start:
                best = max(best, signals[i])
    return best


# ── Intron classification ──────────────────────────────────────────────────────

def _parse_coord(coord: str) -> tuple[str, int, int]:
    """'chr17:64500882-64500986' → (chrom, start, end)."""
    chrom, rest = coord.rsplit(':', 1)
    start_s, end_s = rest.split('-')
    return chrom, int(start_s), int(end_s)


def classify_introns(coret_df: pd.DataFrame,
                     ir_df: pd.DataFrame,
                     fdr_threshold: float,
                     ir_threshold: float) -> pd.DataFrame:
    """
    Classify all introns from ir_df using co-retention significance from coret_df.

    Returns DataFrame with columns:
      coord, chrom, intron_start, intron_end, strand, ir_ratio, intron_class
    """
    # Identify introns in any significant pair
    sig_mask = coret_df['fdr'] < fdr_threshold
    co_coords: set = set()
    for col in ('intron_a', 'intron_b'):
        co_coords.update(coret_df.loc[sig_mask, col].values)

    # Build intron table from IR events (unique introns by coordinate)
    rows = []
    seen: set = set()
    for _, row in ir_df.iterrows():
        chrom = row['chrom']
        s, e = int(row['intron_start']), int(row['intron_end'])
        coord = f"{chrom}:{s}-{e}"
        if coord in seen:
            continue
        seen.add(coord)
        ir_ratio = float(row['ir_ratio'])
        strand = row.get('strand', '+')

        if coord in co_coords:
            cls = 'co_retained'
        elif ir_ratio >= ir_threshold:
            cls = 'indep_retained'
        else:
            cls = 'non_retained'

        rows.append({
            'coord': coord, 'chrom': chrom,
            'intron_start': s, 'intron_end': e,
            'strand': strand, 'ir_ratio': ir_ratio,
            'intron_class': cls,
        })
    return pd.DataFrame(rows)


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(intron_df: pd.DataFrame,
                    fasta: pyfaidx.Fasta,
                    eclip_index: dict | None,
                    u_window: int = 30) -> pd.DataFrame:
    """Add ss5_score, u_richness, ppt_score, and optionally tia1_signal columns."""
    df = intron_df.copy()
    ss5, u_rich, ppt, tia1 = [], [], [], []

    for _, row in df.iterrows():
        chrom = row['chrom']
        s, e = int(row['intron_start']), int(row['intron_end'])
        strand = row['strand']
        intron_len = e - s

        # 5'SS score
        seq9 = extract_5ss_seq(fasta, chrom, s, e, strand)
        ss5.append(score_5ss(seq9) if seq9 else float('nan'))

        # U-richness: skip if intron too short
        if intron_len >= u_window:
            u_rich.append(compute_u_richness(fasta, chrom, s, e, strand, u_window))
        else:
            u_rich.append(None)

        # PPT score
        if intron_len >= u_window:
            ppt.append(compute_ppt_score(fasta, chrom, s, e, strand, u_window))
        else:
            ppt.append(None)

        # TIA1 eCLIP signal
        if eclip_index is not None:
            tia1.append(get_max_eclip_signal(eclip_index, chrom, s, e))
        else:
            tia1.append(None)

    df['ss5_score'] = ss5
    df['u_richness'] = u_rich
    df['ppt_score'] = ppt
    if eclip_index is not None:
        df['tia1_signal'] = tia1

    return df


# ── Statistics ─────────────────────────────────────────────────────────────────

def _bh_fdr(pvals: list[float]) -> list[float]:
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


def run_stats(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U tests between all pairwise group combos for each metric."""
    groups = ['co_retained', 'indep_retained', 'non_retained']
    combos = [
        ('co_retained', 'indep_retained'),
        ('co_retained', 'non_retained'),
        ('indep_retained', 'non_retained'),
    ]
    metric_cols = [c for c in ('ss5_score', 'u_richness', 'ppt_score', 'tia1_signal')
                   if c in metrics_df.columns]

    rows = []
    for metric in metric_cols:
        sub = metrics_df[['intron_class', metric]].dropna()
        group_data = {g: sub.loc[sub['intron_class'] == g, metric].values
                      for g in groups}
        for g1, g2 in combos:
            d1, d2 = group_data[g1], group_data[g2]
            if len(d1) < 3 or len(d2) < 3:
                stat, pval = float('nan'), float('nan')
            else:
                stat, pval = stats.mannwhitneyu(d1, d2, alternative='two-sided')
            rows.append({
                'metric': metric,
                'group1': g1, 'group2': g2,
                'n1': len(d1), 'n2': len(d2),
                'median1': float(np.median(d1)) if len(d1) else float('nan'),
                'median2': float(np.median(d2)) if len(d2) else float('nan'),
                'mw_stat': stat, 'pvalue': pval,
            })

    stats_df = pd.DataFrame(rows)
    if len(stats_df):
        stats_df['fdr'] = _bh_fdr(stats_df['pvalue'].fillna(1.0).tolist())
    return stats_df


# ── Plotting ───────────────────────────────────────────────────────────────────

_CLASS_ORDER = ['co_retained', 'indep_retained', 'non_retained']
_CLASS_COLORS = ['coral', 'steelblue', 'lightgrey']
_CLASS_LABELS = ['Co-retained', 'Indep. retained', 'Non-retained']

_METRIC_LABELS = {
    'ss5_score':   "5'SS PWM score (log-odds)",
    'u_richness':  'U-richness (T fraction, first 30bp)',
    'ppt_score':   'PPT pyrimidine fraction (last 30bp)',
    'tia1_signal': 'TIA1 eCLIP signal (max within 200bp)',
}


def _add_significance(ax, x1: int, x2: int, y: float, pval: float, h: float = 0.02):
    """Draw bracket + p-value annotation between two box positions."""
    if np.isnan(pval):
        return
    if pval < 0.001:
        label = f'p={pval:.1e}'
    elif pval < 0.05:
        label = f'p={pval:.3f}'
    else:
        return  # not significant — skip bracket
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c='black')
    ax.text((x1 + x2) / 2, y + h, label, ha='center', va='bottom', fontsize=7)


def plot_boxplots(metrics_df: pd.DataFrame, stats_df: pd.DataFrame,
                  outdir: str, sample_name: str) -> None:
    metric_cols = [c for c in ('ss5_score', 'u_richness', 'ppt_score', 'tia1_signal')
                   if c in metrics_df.columns]
    n = len(metric_cols)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        data = [
            metrics_df.loc[metrics_df['intron_class'] == g, metric].dropna().values
            for g in _CLASS_ORDER
        ]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        showfliers=True, flierprops=dict(marker='.', ms=2, alpha=0.4))
        for patch, color in zip(bp['boxes'], _CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(_CLASS_LABELS, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(_METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_title(metric.replace('_', ' '), fontsize=9)

        # Add significance brackets for pairwise FDR < 0.05
        metric_stats = stats_df[stats_df['metric'] == metric]
        combo_positions = {
            ('co_retained', 'indep_retained'): (1, 2),
            ('co_retained', 'non_retained'):   (1, 3),
            ('indep_retained', 'non_retained'): (2, 3),
        }
        y_data = [v for d in data for v in d]
        ymax = np.nanpercentile(y_data, 99) if len(y_data) else 0
        yrange = ymax - (np.nanpercentile(y_data, 1) if len(y_data) else 0)
        y_offset = ymax + 0.03 * yrange

        for (g1, g2), (x1, x2) in combo_positions.items():
            row = metric_stats[(metric_stats['group1'] == g1) &
                               (metric_stats['group2'] == g2)]
            if len(row):
                pval = row.iloc[0]['fdr']
                _add_significance(ax, x1, x2, y_offset, pval, h=0.015 * yrange)
                y_offset += 0.08 * yrange

        # Add n= labels
        for xi, (group, d) in enumerate(zip(_CLASS_LABELS, data), 1):
            ax.text(xi, ax.get_ylim()[0], f'n={len(d)}',
                    ha='center', va='top', fontsize=7, color='grey')

    fig_title = f'{sample_name}\nSplice site features by intron retention class'
    plt.suptitle(fig_title, fontsize=10, y=1.02)
    plt.tight_layout()

    fig_dir = os.path.join(outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f'{sample_name}_splice_site_boxplots.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coretention', required=True,
                        help='*_coretention.tsv from co-retention analysis')
    parser.add_argument('--ir-events', required=True,
                        help='*_ir_events.tsv with chrom/intron_start/intron_end/strand/ir_ratio')
    parser.add_argument('--genome', required=True,
                        help='GRCh38 FASTA (must have .fai index)')
    parser.add_argument('--eclip-dir', default=None,
                        help='Dir with TIA1.bed (narrowPeak-format eCLIP). Optional.')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--fdr-threshold', type=float, default=0.05,
                        help='Co-retention FDR cutoff for co_retained class')
    parser.add_argument('--ir-threshold', type=float, default=0.05,
                        help='IR ratio cutoff for indep_retained vs non_retained')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sample_name = os.path.basename(args.coretention).replace('_coretention.tsv', '')
    print(f"Sample: {sample_name}")

    # Load inputs
    print(f"Loading co-retention: {args.coretention}")
    coret_df = pd.read_csv(args.coretention, sep='\t')
    n_sig = (coret_df['fdr'] < args.fdr_threshold).sum()
    print(f"  {len(coret_df)} pairs, {n_sig} significant (FDR<{args.fdr_threshold})")

    print(f"Loading IR events: {args.ir_events}")
    ir_df = pd.read_csv(args.ir_events, sep='\t')
    print(f"  {len(ir_df)} intron records")

    # Classify introns
    print("Classifying introns...")
    intron_df = classify_introns(coret_df, ir_df, args.fdr_threshold, args.ir_threshold)
    counts = intron_df['intron_class'].value_counts()
    for cls in ('co_retained', 'indep_retained', 'non_retained'):
        print(f"  {cls:<20}: {counts.get(cls, 0):>6}")

    # Load genome
    print(f"Loading genome FASTA: {args.genome}")
    fasta = pyfaidx.Fasta(args.genome, as_raw=False)

    # Load TIA1 eCLIP (optional)
    eclip_index = None
    if args.eclip_dir:
        tia1_bed = os.path.join(args.eclip_dir, 'TIA1.bed')
        if os.path.isfile(tia1_bed):
            print(f"Loading TIA1 eCLIP: {tia1_bed}")
            eclip_index = load_eclip_signal(tia1_bed)
            n_peaks = sum(len(v[0]) for v in eclip_index.values())
            print(f"  {n_peaks} TIA1 peaks loaded")
        else:
            print(f"  TIA1.bed not found in {args.eclip_dir} — skipping eCLIP metric")

    # Compute metrics
    print("Computing sequence metrics...")
    metrics_df = compute_metrics(intron_df, fasta, eclip_index)
    fasta.close()

    n_valid_5ss = metrics_df['ss5_score'].notna().sum()
    print(f"  5'SS scores computed: {n_valid_5ss}/{len(metrics_df)}")
    for metric in ('u_richness', 'ppt_score'):
        n_v = metrics_df[metric].notna().sum()
        print(f"  {metric}: {n_v}/{len(metrics_df)} valid")
    if 'tia1_signal' in metrics_df.columns:
        n_bound = (metrics_df['tia1_signal'] > 0).sum()
        print(f"  TIA1 signal > 0: {n_bound}/{len(metrics_df)}")

    # Group medians
    print("\nGroup medians:")
    metric_cols = [c for c in ('ss5_score', 'u_richness', 'ppt_score', 'tia1_signal')
                   if c in metrics_df.columns]
    for cls in ('co_retained', 'indep_retained', 'non_retained'):
        sub = metrics_df[metrics_df['intron_class'] == cls]
        vals = [f"{sub[m].median():.4f}" if sub[m].notna().any() else 'NA'
                for m in metric_cols]
        print(f"  {cls:<20}: {dict(zip(metric_cols, vals))}")

    # Statistics
    print("\nRunning Mann-Whitney tests...")
    stats_df = run_stats(metrics_df)
    sig = stats_df[stats_df['fdr'] < 0.05] if len(stats_df) else pd.DataFrame()
    print(f"  Significant comparisons (FDR<0.05): {len(sig)}/{len(stats_df)}")
    if len(sig):
        print(sig[['metric', 'group1', 'group2', 'median1', 'median2',
                   'pvalue', 'fdr']].to_string(index=False))

    # Save outputs
    out_metrics = os.path.join(args.outdir, f'{sample_name}_splice_site_metrics.tsv')
    metrics_df.to_csv(out_metrics, sep='\t', index=False)
    print(f"\nSaved metrics: {out_metrics}")

    out_stats = os.path.join(args.outdir, f'{sample_name}_splice_site_stats.tsv')
    stats_df.to_csv(out_stats, sep='\t', index=False)
    print(f"Saved stats:   {out_stats}")

    plot_boxplots(metrics_df, stats_df, args.outdir, sample_name)


if __name__ == '__main__':
    main()

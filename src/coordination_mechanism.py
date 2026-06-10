"""
coordination_mechanism.py — Three mechanistic tests for co-retention coordination.

Analysis 1 — Intervening exon length:
  For adjacent intron pairs, compute the exon gap (intron_b_start − intron_a_end) and
  compare across: co_retained (pair FDR<0.05), both_ir (non-sig but both IR≥threshold),
  no_ir (non-sig, both IR<threshold). Mann-Whitney tests.

Analysis 2 — TIA1 dual binding:
  For each intron pair, check whether TIA1 eCLIP has a peak within 200 bp of the 5'SS
  of BOTH introns simultaneously. Compare co_retained vs non_co_ir pairs via Fisher's
  exact test. Only runs when --eclip-dir is provided (K562 / HepG2 only).

Analysis 3 — Directional asymmetry:
  From the asymmetry file, filter pairs with raw asymmetry_pvalue < --asym-pval-threshold.
  Classify each direction as 'upstream_retained' or 'downstream_retained' relative to
  Pol II transcription order (requires strand from --ir-events). Outputs per-sample
  direction counts for cross-sample aggregation in the SLURM script.

Usage:
    python src/coordination_mechanism.py \\
        --coretention  results/SGNex_K562_directRNA_replicate4_run1/..._coretention.tsv \\
        --ir-events    results/SGNex_K562_directRNA_replicate4_run1/..._ir_events.tsv \\
        --asymmetry    results/SGNex_K562_directRNA_replicate4_run1/..._asymmetry.tsv \\
        --eclip-dir    data/encode_eclip/K562/ \\
        --outdir       results/mechanistic/coordination/ \\
        --cell-line    K562 \\
        [--fdr-threshold 0.05] [--ir-threshold 0.05] [--asym-pval-threshold 0.05]
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
from scipy import stats
from scipy.stats import binomtest


# ── Shared helpers ─────────────────────────────────────────────────────────────

def parse_coord(coord: str) -> tuple[str, int, int]:
    """'chr17:64500882-64500986' → (chrom, start, end)."""
    chrom, rest = coord.rsplit(':', 1)
    s, e = rest.split('-')
    return chrom, int(s), int(e)


def load_eclip_index(bed_path: str) -> dict:
    """
    Load narrowPeak BED → {chrom: (sorted_starts, ends)}.
    Uses parallel sorted arrays for bisect-based overlap.
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
                raw[parts[0]].append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    index = {}
    for chrom, peaks in raw.items():
        peaks.sort()
        index[chrom] = ([p[0] for p in peaks], [p[1] for p in peaks])
    return index


def has_peak_near(index: dict, chrom: str, pos: int, window: int = 200) -> bool:
    """True if any eCLIP peak overlaps [pos−window, pos+window)."""
    if chrom not in index:
        return False
    starts, ends = index[chrom]
    q_start, q_end = pos - window, pos + window
    hi = bisect_left(starts, q_end)
    for i in range(hi - 1, -1, -1):
        if starts[i] < q_start - window:
            break
        if ends[i] > q_start:
            return True
    return False


def build_strand_lookup(ir_df: pd.DataFrame) -> dict:
    """
    Returns {(chrom, intron_start, intron_end): strand}.
    Uses first occurrence when duplicates exist.
    """
    lookup = {}
    for _, row in ir_df.iterrows():
        key = (row['chrom'], int(row['intron_start']), int(row['intron_end']))
        if key not in lookup:
            lookup[key] = row.get('strand', '+')
    return lookup


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


# ── Analysis 1: Intervening exon length ───────────────────────────────────────

def analysis_exon_length(coret_df: pd.DataFrame,
                         fdr_threshold: float,
                         ir_threshold: float) -> pd.DataFrame:
    """
    For adjacent pairs, compute exon gap and assign group:
      co_retained, both_ir, no_ir, mixed_ir.
    Returns DataFrame with columns: group, exon_len, chrom, intron_a, intron_b, fdr.
    """
    adj = coret_df[coret_df['adjacent'] == True].copy()
    rows = []
    for _, r in adj.iterrows():
        try:
            chrom_a, _, end_a = parse_coord(r['intron_a'])
            _, start_b, _ = parse_coord(r['intron_b'])
        except Exception:
            continue
        exon_len = start_b - end_a
        if exon_len < 0:
            continue  # should not happen; intron_a always has lower coords

        fdr_val = float(r['fdr'])
        ir_a = float(r['ir_ratio_a'])
        ir_b = float(r['ir_ratio_b'])

        if fdr_val < fdr_threshold:
            group = 'co_retained'
        elif ir_a >= ir_threshold and ir_b >= ir_threshold:
            group = 'both_ir'
        elif ir_a < ir_threshold and ir_b < ir_threshold:
            group = 'no_ir'
        else:
            group = 'mixed_ir'

        rows.append({'group': group, 'exon_len': exon_len,
                     'chrom': chrom_a, 'intron_a': r['intron_a'],
                     'intron_b': r['intron_b'], 'fdr': fdr_val})
    return pd.DataFrame(rows)


def stats_exon_length(exon_df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U tests between all pairwise group combos for exon_len."""
    target_groups = ['co_retained', 'both_ir', 'no_ir']
    gd = {g: exon_df.loc[exon_df['group'] == g, 'exon_len'].values
          for g in target_groups}
    combos = [('co_retained', 'both_ir'),
              ('co_retained', 'no_ir'),
              ('both_ir', 'no_ir')]
    rows = []
    for g1, g2 in combos:
        d1, d2 = gd[g1], gd[g2]
        if len(d1) >= 3 and len(d2) >= 3:
            stat, pval = stats.mannwhitneyu(d1, d2, alternative='two-sided')
        else:
            stat, pval = float('nan'), float('nan')
        rows.append({'group1': g1, 'group2': g2,
                     'n1': len(d1), 'n2': len(d2),
                     'median1': float(np.median(d1)) if len(d1) else float('nan'),
                     'median2': float(np.median(d2)) if len(d2) else float('nan'),
                     'mw_stat': stat, 'pvalue': pval})
    df = pd.DataFrame(rows)
    if len(df):
        df['fdr'] = _bh_fdr(df['pvalue'].fillna(1.0).tolist())
    return df


# ── Analysis 2: TIA1 dual binding ─────────────────────────────────────────────

def _get_5ss_pos(chrom: str, intron_start: int, intron_end: int,
                 strand_lookup: dict) -> int:
    """Return genomic position of the 5'SS for the intron."""
    strand = strand_lookup.get((chrom, intron_start, intron_end), '+')
    return intron_start if strand == '+' else intron_end


def analysis_tia1_cobinding(coret_df: pd.DataFrame,
                             strand_lookup: dict,
                             eclip_index: dict,
                             fdr_threshold: float,
                             ir_threshold: float,
                             window: int = 200) -> dict:
    """
    For each pair, check TIA1 peak within `window` bp of 5'SS of BOTH introns.
    Groups: co_retained (fdr<threshold) vs non_co_ir (not sig, both IR≥threshold).
    Returns dict with per-group counts and Fisher stats.
    """
    groups = {'co_retained': {'dual': 0, 'total': 0},
              'non_co_ir':   {'dual': 0, 'total': 0}}

    for _, r in coret_df.iterrows():
        fdr_val = float(r['fdr'])
        ir_a = float(r['ir_ratio_a'])
        ir_b = float(r['ir_ratio_b'])

        if fdr_val < fdr_threshold:
            grp = 'co_retained'
        elif ir_a >= ir_threshold and ir_b >= ir_threshold:
            grp = 'non_co_ir'
        else:
            continue

        try:
            chrom_a, start_a, end_a = parse_coord(r['intron_a'])
            chrom_b, start_b, end_b = parse_coord(r['intron_b'])
        except Exception:
            continue

        ss_a = _get_5ss_pos(chrom_a, start_a, end_a, strand_lookup)
        ss_b = _get_5ss_pos(chrom_b, start_b, end_b, strand_lookup)

        bound_a = has_peak_near(eclip_index, chrom_a, ss_a, window)
        bound_b = has_peak_near(eclip_index, chrom_b, ss_b, window)
        dual = bound_a and bound_b

        groups[grp]['total'] += 1
        if dual:
            groups[grp]['dual'] += 1

    co = groups['co_retained']
    nc = groups['non_co_ir']

    result = {
        'co_retained_dual': co['dual'],
        'co_retained_total': co['total'],
        'co_retained_frac': co['dual'] / co['total'] if co['total'] else float('nan'),
        'non_co_ir_dual': nc['dual'],
        'non_co_ir_total': nc['total'],
        'non_co_ir_frac': nc['dual'] / nc['total'] if nc['total'] else float('nan'),
    }

    if co['total'] > 0 and nc['total'] > 0:
        table = [
            [co['dual'],         co['total'] - co['dual']],
            [nc['dual'],         nc['total'] - nc['dual']],
        ]
        or_, pval = stats.fisher_exact(table, alternative='greater')
    else:
        or_, pval = float('nan'), float('nan')

    result['fisher_or'] = or_
    result['fisher_pvalue'] = pval
    return result


# ── Analysis 3: Directional asymmetry ─────────────────────────────────────────

def analysis_directional(asym_df: pd.DataFrame,
                          strand_lookup: dict,
                          asym_pval_threshold: float = 0.05) -> dict:
    """
    Filter to pairs with raw asymmetry_pvalue < threshold.
    Map direction to upstream/downstream using strand.
    Returns dict: n_upstream_retained, n_downstream_retained, n_symmetric, n_no_strand.
    """
    if len(asym_df) == 0 or 'asymmetry_pvalue' not in asym_df.columns:
        return {'n_upstream_retained': 0, 'n_downstream_retained': 0,
                'n_symmetric': 0, 'n_no_strand': 0, 'n_total': 0}

    filtered = asym_df[asym_df['asymmetry_pvalue'] < asym_pval_threshold]
    n_upstream = n_downstream = n_sym = n_no_strand = 0

    for _, r in filtered.iterrows():
        direction = r.get('direction', 'symmetric')
        if direction == 'symmetric':
            n_sym += 1
            continue

        try:
            chrom_a, start_a, end_a = parse_coord(r['intron_a'])
        except Exception:
            n_no_strand += 1
            continue

        strand = strand_lookup.get((chrom_a, start_a, end_a), None)
        if strand is None:
            # Try intron_b
            try:
                chrom_b, start_b, end_b = parse_coord(r['intron_b'])
                strand = strand_lookup.get((chrom_b, start_b, end_b), None)
            except Exception:
                pass
        if strand is None:
            n_no_strand += 1
            continue

        # intron_a always has lower genomic coords
        # + strand: a = upstream, b = downstream
        # - strand: b = upstream (higher coords = transcribed first), a = downstream
        if strand == '+':
            a_is_upstream = True
        else:
            a_is_upstream = False  # on - strand, a (lower coords) is downstream

        if direction == 'A_retained_preferentially':
            if a_is_upstream:
                n_upstream += 1
            else:
                n_downstream += 1
        elif direction == 'B_retained_preferentially':
            if a_is_upstream:
                n_downstream += 1
            else:
                n_upstream += 1
        else:
            n_sym += 1

    n_total = n_upstream + n_downstream + n_sym + n_no_strand
    return {
        'n_upstream_retained': n_upstream,
        'n_downstream_retained': n_downstream,
        'n_symmetric': n_sym,
        'n_no_strand': n_no_strand,
        'n_total': n_total,
    }


# ── Plotting (per-sample) ──────────────────────────────────────────────────────

def plot_sample(exon_df: pd.DataFrame,
                tia1_result: dict | None,
                direction_counts: dict,
                outdir: str,
                sample_name: str) -> None:

    n_panels = 1 + (1 if tia1_result else 0) + 1  # exon + tia1 + direction
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # Panel 1: exon length boxplot
    ax = axes[ax_idx]; ax_idx += 1
    groups_order = ['co_retained', 'both_ir', 'no_ir']
    group_labels = ['Co-retained\n(FDR<0.05)', 'Both IR\n(non-sig)', 'No IR\n(non-sig)']
    group_colors = ['coral', 'steelblue', 'lightgrey']
    data = [exon_df.loc[exon_df['group'] == g, 'exon_len'].values
            for g in groups_order]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    showfliers=True, flierprops=dict(marker='.', ms=2, alpha=0.4))
    for patch, color in zip(bp['boxes'], group_colors):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel('Intervening exon length (bp)', fontsize=9)
    ax.set_title('Exon between co-retained introns', fontsize=9)
    for xi, d in enumerate(data, 1):
        ax.text(xi, ax.get_ylim()[0], f'n={len(d)}',
                ha='center', va='top', fontsize=7, color='grey')

    # Panel 2: TIA1 dual binding bar (if available)
    if tia1_result:
        ax = axes[ax_idx]; ax_idx += 1
        labels = ['Co-retained', 'Non-co IR']
        fracs = [
            tia1_result.get('co_retained_frac', 0) * 100,
            tia1_result.get('non_co_ir_frac', 0) * 100,
        ]
        ns = [tia1_result.get('co_retained_total', 0),
              tia1_result.get('non_co_ir_total', 0)]
        bars = ax.bar(labels, fracs, color=['coral', 'steelblue'], alpha=0.8)
        for bar, n, frac in zip(bars, ns, fracs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{frac:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=8)
        pval = tia1_result.get('fisher_pvalue', float('nan'))
        or_ = tia1_result.get('fisher_or', float('nan'))
        pstr = f'OR={or_:.2f}, p={pval:.2e}' if not np.isnan(pval) else ''
        ax.set_ylabel('% pairs with dual TIA1 at 5\'SS (±200bp)', fontsize=9)
        ax.set_title(f'TIA1 dual binding\n{pstr}', fontsize=9)

    # Panel 3: direction bar
    ax = axes[ax_idx]
    n_up = direction_counts.get('n_upstream_retained', 0)
    n_dn = direction_counts.get('n_downstream_retained', 0)
    bars = ax.bar(['Upstream\nretained', 'Downstream\nretained'],
                  [n_up, n_dn], color=['darkorange', 'steelblue'], alpha=0.8)
    for bar, val in zip(bars, [n_up, n_dn]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, str(val),
                ha='center', va='bottom', fontsize=9)
    total_dir = n_up + n_dn
    if total_dir > 0:
        res = binomtest(n_up, total_dir, p=0.5, alternative='greater')
        ax.set_title(f'Directional asymmetry\n(asym p<0.05 pairs)\n'
                     f'binom p={res.pvalue:.3f}', fontsize=9)
    else:
        ax.set_title('Directional asymmetry\n(no data)', fontsize=9)
    ax.set_ylabel('Count of pairs', fontsize=9)

    fig.suptitle(sample_name, fontsize=10, y=1.01)
    plt.tight_layout()
    fig_dir = os.path.join(outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, f'{sample_name}_coordination.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coretention', required=True)
    parser.add_argument('--ir-events', required=True)
    parser.add_argument('--asymmetry', default=None,
                        help='*_asymmetry.tsv (optional — skips analysis 3 if absent)')
    parser.add_argument('--eclip-dir', default=None,
                        help='Dir with TIA1.bed for this cell line (optional)')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--cell-line', default='unknown')
    parser.add_argument('--fdr-threshold', type=float, default=0.05)
    parser.add_argument('--ir-threshold', type=float, default=0.05)
    parser.add_argument('--asym-pval-threshold', type=float, default=0.05,
                        help='Raw binomial p-value cutoff for directional asymmetry')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sample_name = os.path.basename(args.coretention).replace('_coretention.tsv', '')
    print(f"Sample: {sample_name}  ({args.cell_line})")

    # Load data
    coret_df = pd.read_csv(args.coretention, sep='\t')
    ir_df = pd.read_csv(args.ir_events, sep='\t')
    n_sig = (coret_df['fdr'] < args.fdr_threshold).sum()
    print(f"  {len(coret_df)} pairs, {n_sig} significant (FDR<{args.fdr_threshold})")

    strand_lookup = build_strand_lookup(ir_df)
    print(f"  Strand lookup: {len(strand_lookup)} introns")

    # ── Analysis 1: exon length ──────────────────────────────────────────────
    print("\n[1] Intervening exon length...")
    exon_df = analysis_exon_length(coret_df, args.fdr_threshold, args.ir_threshold)
    counts = exon_df['group'].value_counts()
    for g in ('co_retained', 'both_ir', 'no_ir', 'mixed_ir'):
        n = counts.get(g, 0)
        if n:
            med = exon_df.loc[exon_df['group'] == g, 'exon_len'].median()
            print(f"  {g:<15}: n={n:>5}, median={med:.0f}bp")

    exon_stats_df = stats_exon_length(exon_df)
    sig_el = exon_stats_df[exon_stats_df.get('fdr', pd.Series([])) < 0.05] \
        if 'fdr' in exon_stats_df.columns else pd.DataFrame()
    print(f"  Significant Mann-Whitney comparisons: {len(sig_el)}/{len(exon_stats_df)}")
    if len(sig_el):
        print(sig_el[['group1', 'group2', 'n1', 'n2', 'median1', 'median2',
                       'pvalue', 'fdr']].to_string(index=False))

    out_exon = os.path.join(args.outdir, f'{sample_name}_exon_length.tsv')
    exon_df.to_csv(out_exon, sep='\t', index=False)

    out_exon_stats = os.path.join(args.outdir, f'{sample_name}_exon_length_stats.tsv')
    exon_stats_df.to_csv(out_exon_stats, sep='\t', index=False)

    # ── Analysis 2: TIA1 dual binding ────────────────────────────────────────
    tia1_result = None
    if args.eclip_dir:
        tia1_bed = os.path.join(args.eclip_dir, 'TIA1.bed')
        if os.path.isfile(tia1_bed):
            print(f"\n[2] TIA1 dual binding ({args.cell_line})...")
            eclip_index = load_eclip_index(tia1_bed)
            n_peaks = sum(len(v[0]) for v in eclip_index.values())
            print(f"  {n_peaks} TIA1 peaks")
            tia1_result = analysis_tia1_cobinding(
                coret_df, strand_lookup, eclip_index,
                args.fdr_threshold, args.ir_threshold)
            print(f"  co_retained:  {tia1_result['co_retained_dual']}"
                  f"/{tia1_result['co_retained_total']}"
                  f" ({tia1_result['co_retained_frac']*100:.1f}%) dual TIA1")
            print(f"  non_co_ir:    {tia1_result['non_co_ir_dual']}"
                  f"/{tia1_result['non_co_ir_total']}"
                  f" ({tia1_result['non_co_ir_frac']*100:.1f}%) dual TIA1")
            print(f"  Fisher OR={tia1_result['fisher_or']:.3f}"
                  f"  p={tia1_result['fisher_pvalue']:.3e}")
            out_tia1 = os.path.join(args.outdir, f'{sample_name}_tia1_cobinding.tsv')
            tia1_df = pd.DataFrame([{**tia1_result, 'sample': sample_name,
                                      'cell_line': args.cell_line}])
            tia1_df.to_csv(out_tia1, sep='\t', index=False)
        else:
            print(f"\n[2] TIA1.bed not found in {args.eclip_dir} — skipping")
    else:
        print("\n[2] No --eclip-dir — skipping TIA1 analysis")

    # ── Analysis 3: directional asymmetry ────────────────────────────────────
    direction_counts: dict = {
        'n_upstream_retained': 0, 'n_downstream_retained': 0,
        'n_symmetric': 0, 'n_no_strand': 0, 'n_total': 0,
    }
    if args.asymmetry and os.path.isfile(args.asymmetry):
        print(f"\n[3] Directional asymmetry (raw p<{args.asym_pval_threshold})...")
        asym_df = pd.read_csv(args.asymmetry, sep='\t')
        if len(asym_df) > 0 and 'asymmetry_pvalue' in asym_df.columns:
            direction_counts = analysis_directional(
                asym_df, strand_lookup, args.asym_pval_threshold)
            n_up = direction_counts['n_upstream_retained']
            n_dn = direction_counts['n_downstream_retained']
            n_tot = n_up + n_dn
            print(f"  upstream_retained:   {n_up}")
            print(f"  downstream_retained: {n_dn}")
            print(f"  symmetric:           {direction_counts['n_symmetric']}")
            print(f"  no_strand_info:      {direction_counts['n_no_strand']}")
            if n_tot > 0:
                res = binomtest(n_up, n_tot, p=0.5, alternative='greater')
                frac = n_up / n_tot * 100
                print(f"  Upstream fraction: {frac:.1f}%  binom p={res.pvalue:.4f}")
        else:
            print("  Empty or malformed asymmetry file — skipping")
    else:
        print("\n[3] No --asymmetry file — skipping")

    out_dir_counts = os.path.join(args.outdir, f'{sample_name}_direction_counts.tsv')
    pd.DataFrame([{**direction_counts, 'sample': sample_name,
                   'cell_line': args.cell_line}]).to_csv(
        out_dir_counts, sep='\t', index=False)

    # ── Per-sample figure ────────────────────────────────────────────────────
    plot_sample(exon_df, tia1_result, direction_counts, args.outdir, sample_name)

    print(f"\nOutputs → {args.outdir}")


if __name__ == '__main__':
    main()

"""
asymmetry_test.py — Test for directional splicing order dependency in co-retained intron pairs.

For each significant co-retention pair, tests whether the off-diagonal cells of the 2x2
contingency table are asymmetric using a binomial test:

  H0: P(A_retained, B_spliced) = P(A_spliced, B_retained) = 0.5
  Test: binomtest(min(a_ret_b_spl, a_spl_b_ret), a_ret_b_spl + a_spl_b_ret, 0.5)

A significant result implies that one intron is preferentially retained when the other is
spliced — consistent with a directional splicing dependency (A must splice before B, or vice versa).

Usage:
    python src/asymmetry_test.py \
        --coretention results/pilot_v2/SGNex_A549_directRNA_replicate1_run1_coretention.tsv \
        --outdir results/pilot_v2/ \
        --fdr-threshold 0.05
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest


def run_asymmetry_test(df: pd.DataFrame, fdr_threshold: float = 0.05) -> pd.DataFrame:
    sig = df[df['fdr'] < fdr_threshold].copy()
    if len(sig) == 0:
        return pd.DataFrame()

    rows = []
    for _, row in sig.iterrows():
        a = int(row['a_ret_b_spl'])
        b = int(row['a_spl_b_ret'])
        total_off_diag = a + b
        if total_off_diag == 0:
            continue

        result = binomtest(min(a, b), total_off_diag, p=0.5, alternative='less')
        pval = result.pvalue

        # Direction: which intron is more often retained when the other splices
        if a > b:
            direction = "A_retained_preferentially"
        elif b > a:
            direction = "B_retained_preferentially"
        else:
            direction = "symmetric"

        rows.append({
            'gene_name': row['gene_name'],
            'gene_id': row['gene_id'],
            'intron_a': row['intron_a'],
            'intron_b': row['intron_b'],
            'intron_a_index': row['intron_a_index'],
            'intron_b_index': row['intron_b_index'],
            'adjacent': row['adjacent'],
            'a_ret_b_spl': a,
            'a_spl_b_ret': b,
            'total_off_diagonal': total_off_diag,
            'asymmetry_pvalue': pval,
            'direction': direction,
            'phi_coefficient': row['phi_coefficient'],
            'coretention_fdr': row['fdr'],
        })

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    # FDR correct asymmetry p-values
    pvals = result_df['asymmetry_pvalue'].values
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    fdr = [0.0] * n
    cummin = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, pval = indexed[rank_minus_1]
        adjusted = pval * n / (rank_minus_1 + 1)
        cummin = min(cummin, adjusted, 1.0)
        fdr[orig_idx] = cummin
    result_df['asymmetry_fdr'] = fdr

    return result_df.sort_values('asymmetry_pvalue')


def plot_asymmetry(result_df: pd.DataFrame, outdir: str, sample_name: str):
    if len(result_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: scatter of off-diagonal counts
    ax = axes[0]
    sig_asym = result_df[result_df['asymmetry_fdr'] < 0.05]
    ax.scatter(result_df['a_ret_b_spl'], result_df['a_spl_b_ret'],
               alpha=0.5, s=20, color='steelblue', label='Symmetric')
    if len(sig_asym) > 0:
        ax.scatter(sig_asym['a_ret_b_spl'], sig_asym['a_spl_b_ret'],
                   alpha=0.8, s=40, color='coral', label=f'Asymmetric FDR<0.05 (n={len(sig_asym)})')
    lim = max(result_df['a_ret_b_spl'].max(), result_df['a_spl_b_ret'].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Reads: A retained, B spliced')
    ax.set_ylabel('Reads: A spliced, B retained')
    ax.set_title('Off-diagonal asymmetry\n(deviation from diagonal = directional dependency)')
    ax.legend(fontsize=8)

    # Panel B: asymmetry p-value distribution
    ax = axes[1]
    ax.hist(result_df['asymmetry_pvalue'], bins=20, color='steelblue', edgecolor='none')
    ax.axvline(0.05, color='red', linestyle='--', label='p=0.05')
    ax.set_xlabel('Asymmetry binomial p-value')
    ax.set_ylabel('Count')
    ax.set_title(f'Splicing order asymmetry\n({len(sig_asym)} of {len(result_df)} pairs significant)')
    ax.legend()

    plt.tight_layout()
    out = os.path.join(outdir, 'figures', f'{sample_name}_asymmetry.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved asymmetry plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Splicing order asymmetry test")
    parser.add_argument("--coretention", required=True, help="Co-retention TSV from run_pilot.py")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sample_name = os.path.basename(args.coretention).replace('_coretention.tsv', '')

    print(f"Loading co-retention results: {args.coretention}")
    df = pd.read_csv(args.coretention, sep='\t')
    print(f"  {len(df)} pairs total, {(df['fdr'] < args.fdr_threshold).sum()} significant (FDR<{args.fdr_threshold})")

    print("\nRunning asymmetry test...")
    result_df = run_asymmetry_test(df, fdr_threshold=args.fdr_threshold)

    if len(result_df) == 0:
        print("  No significant co-retention pairs to test.")
        out_tsv = os.path.join(args.outdir, f'{sample_name}_asymmetry.tsv')
        pd.DataFrame(columns=['gene_name', 'gene_id', 'intron_a', 'intron_b',
                               'intron_a_index', 'intron_b_index', 'adjacent',
                               'a_ret_b_spl', 'a_spl_b_ret', 'total_off_diagonal',
                               'asymmetry_pvalue', 'direction', 'phi_coefficient',
                               'coretention_fdr', 'asymmetry_fdr']).to_csv(out_tsv, sep='\t', index=False)
        return

    n_asym = (result_df['asymmetry_fdr'] < 0.05).sum()
    print(f"  Pairs tested: {len(result_df)}")
    print(f"  Significant asymmetry (FDR<0.05): {n_asym}")
    if n_asym > 0:
        print("\n  Top asymmetric pairs:")
        cols = ['gene_name', 'intron_a_index', 'intron_b_index', 'a_ret_b_spl', 'a_spl_b_ret', 'direction', 'asymmetry_fdr']
        print(result_df[result_df['asymmetry_fdr'] < 0.05][cols].to_string(index=False))

    out_tsv = os.path.join(args.outdir, f'{sample_name}_asymmetry.tsv')
    result_df.to_csv(out_tsv, sep='\t', index=False)
    print(f"\nSaved: {out_tsv}")

    plot_asymmetry(result_df, args.outdir, sample_name)


if __name__ == "__main__":
    main()

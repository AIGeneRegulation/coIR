"""
rbp_motifs.py — RBP motif enrichment at co-retained vs non-co-retained IR intron boundaries.

For each intron, extracts four 30bp sequence windows:
  - 5p_intron:  first 30bp of the intron (downstream of 5' splice site)
  - 3p_intron:  last 30bp of the intron (upstream of 3' splice site)
  - 5p_exon:    30bp of upstream exon (exonic, ending at 5' SS)
  - 3p_exon:    30bp of downstream exon (exonic, starting at 3' SS)

Computes k-mer frequencies (k=4,5,6) and compares co-retained vs IR-only introns
with Fisher's exact test + BH FDR. Maps significant k-mers to known RBP motifs.

Usage:
    python src/rbp_motifs.py \
        --coretention results/pilot_v2/<sample>_coretention.tsv \
        --ir-events results/pilot_v2/<sample>_ir_events.tsv \
        --genome data/annotations/GRCh38.primary_assembly.genome.fa \
        --outdir results/pilot_v2/ \
        [--fdr-threshold 0.05] [--min-ir-ratio 0.05]
"""

import argparse
import itertools
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact

# Known RBP motifs (DNA alphabet). Degenerate IUPAC: Y=CT, R=AG, W=AT, S=CG, K=GT, M=AC
RBP_MOTIFS = {
    'SRSF1':  ['GGAGGA', 'GAGGAG'],
    'SRSF2':  ['SSNG'],            # degenerate; will expand below
    'SRSF7':  ['GATGAT', 'GATCAT', 'CATGAT', 'CATCAT'],
    'PTBP1':  ['TCCCC', 'CCCCT', 'TCTCT', 'TCCTC'],
    'TIA1':   ['TTTTT', 'TTTTA', 'ATTTT'],
    'hnRNP_A1': ['TAGG', 'TAGGT'],
    'hnRNP_C':  ['TTTTT', 'TTTTTT'],
    'hnRNP_F':  ['GGGG', 'GGGGG'],
    'MBNL1':    ['TGCT', 'TGCC', 'CGCT', 'CGCC'],
    'ELAVL1':   ['TTTAT', 'TATTTT', 'TTATTT'],
    'FUS':      ['GGUG', 'GGTG'],
    'U2AF2':    ['TTTTTT', 'TTTTCT', 'CTTTTT'],
}

WINDOW = 30  # bp extracted at each boundary


def expand_iupac(seq: str) -> list[str]:
    """Expand a degenerate IUPAC sequence to all concrete sequences."""
    iupac = {
        'R': 'AG', 'Y': 'CT', 'S': 'CG', 'W': 'AT',
        'K': 'GT', 'M': 'AC', 'B': 'CGT', 'D': 'AGT',
        'H': 'ACT', 'V': 'ACG', 'N': 'ACGT',
    }
    result = ['']
    for char in seq.upper():
        if char in iupac:
            result = [r + c for r in result for c in iupac[char]]
        else:
            result = [r + char for r in result]
    return result


# Pre-expand all motifs
RBP_MOTIFS_EXPANDED = {
    name: list(set(exp for m in motifs for exp in expand_iupac(m)))
    for name, motifs in RBP_MOTIFS.items()
}


def get_sequence(fasta: pysam.FastaFile, chrom: str, start: int, end: int,
                 strand: str = '+') -> str | None:
    """Fetch genomic sequence, handling chr prefix mismatch."""
    refs = set(fasta.references)
    c = chrom if chrom in refs else ('chr' + chrom if 'chr' + chrom in refs
                                      else chrom.lstrip('chr') if chrom.lstrip('chr') in refs
                                      else None)
    if c is None:
        return None
    start = max(0, start)
    end = min(fasta.get_reference_length(c), end)
    if end <= start:
        return None
    seq = fasta.fetch(c, start, end).upper()
    if strand == '-':
        seq = seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]
    return seq


def extract_boundary_sequences(intron_coord: str, strand: str,
                                fasta: pysam.FastaFile) -> dict[str, str] | None:
    """
    Parse intron_coord ("chrom:start-end") and extract 4 boundary windows.
    start/end are 0-based half-open.
    """
    m = re.match(r'(.+):(\d+)-(\d+)', intron_coord)
    if not m:
        return None
    chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))

    seqs = {}
    # 5' splice site (intron side)
    seqs['5p_intron'] = get_sequence(fasta, chrom, start, start + WINDOW, strand)
    # 3' splice site (intron side)
    seqs['3p_intron'] = get_sequence(fasta, chrom, end - WINDOW, end, strand)
    # upstream exon
    seqs['5p_exon'] = get_sequence(fasta, chrom, start - WINDOW, start, strand)
    # downstream exon
    seqs['3p_exon'] = get_sequence(fasta, chrom, end, end + WINDOW, strand)

    return {k: v for k, v in seqs.items() if v is not None}


def kmer_counts(sequences: list[str], k: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if set(kmer) <= set('ACGT'):
                counts[kmer] += 1
    return dict(counts)


def all_kmers(k: int) -> list[str]:
    return [''.join(p) for p in itertools.product('ACGT', repeat=k)]


def enrichment_test(coret_seqs: list[str], ctrl_seqs: list[str],
                    ks: list[int] = [4, 5, 6]) -> pd.DataFrame:
    rows = []
    for k in ks:
        coret_counts = kmer_counts(coret_seqs, k)
        ctrl_counts = kmer_counts(ctrl_seqs, k)
        total_coret = sum(coret_counts.values())
        total_ctrl = sum(ctrl_counts.values())

        for kmer in all_kmers(k):
            a = coret_counts.get(kmer, 0)
            b = ctrl_counts.get(kmer, 0)
            c = total_coret - a
            d = total_ctrl - b
            if a + b < 5:
                continue
            _, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
            fc = (a / total_coret + 1e-9) / (b / total_ctrl + 1e-9)
            rows.append({'kmer': kmer, 'k': k, 'coret_count': a, 'ctrl_count': b,
                         'coret_freq': a / total_coret, 'ctrl_freq': b / total_ctrl,
                         'fold_change': fc, 'pvalue': pval})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # BH FDR
    pvals = df['pvalue'].values
    n = len(pvals)
    idx = np.argsort(pvals)
    fdr = np.zeros(n)
    cummin = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        i = idx[rank_minus_1]
        fdr[i] = min(cummin, pvals[i] * n / (rank_minus_1 + 1), 1.0)
        cummin = fdr[i]
    df['fdr'] = fdr

    # Map significant k-mers to RBP motifs
    sig_kmers = set(df.loc[df['fdr'] < 0.05, 'kmer'])
    rbp_hits = []
    for kmer in df['kmer']:
        matched = []
        for rbp, motifs in RBP_MOTIFS_EXPANDED.items():
            if any(kmer in m or m in kmer for m in motifs):
                matched.append(rbp)
        rbp_hits.append(', '.join(matched) if matched else '')
    df['rbp_match'] = rbp_hits

    return df.sort_values('pvalue')


def plot_enrichment(enrich_df: pd.DataFrame, outdir: str, sample_name: str,
                    region: str, top_n: int = 25):
    sig = enrich_df[(enrich_df['fdr'] < 0.05)].head(top_n)
    if len(sig) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(sig) * 0.3)))
    colors = ['coral' if r else 'steelblue' for r in sig['rbp_match']]
    ax.barh(range(len(sig)), np.log2(sig['fold_change']), color=colors)
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels([
        f"{row['kmer']} ({row['rbp_match']})" if row['rbp_match'] else row['kmer']
        for _, row in sig.iterrows()
    ], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('log2(fold change, co-retained / IR-only)')
    ax.set_title(f'Enriched k-mers at {region}\n{sample_name}')
    plt.tight_layout()
    out = os.path.join(outdir, 'figures', f'{sample_name}_kmer_{region}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="RBP motif enrichment at co-retained introns")
    parser.add_argument("--coretention", required=True)
    parser.add_argument("--ir-events", required=True)
    parser.add_argument("--genome", required=True, help="FASTA (indexed with .fai)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--min-ir-ratio", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sample_name = os.path.basename(args.coretention).replace('_coretention.tsv', '')

    print("Loading data...")
    coret_df = pd.read_csv(args.coretention, sep='\t')
    ir_df = pd.read_csv(args.ir_events, sep='\t')

    sig_coret = coret_df[
        (coret_df['fdr'] < args.fdr_threshold) & (coret_df['phi_coefficient'] > 0)
    ]

    # Introns in significant co-retained pairs
    coret_intron_coords = set()
    coret_strands = {}
    for _, row in sig_coret.iterrows():
        for col in ('intron_a', 'intron_b'):
            coret_intron_coords.add(row[col])

    # Map strand from IR events (using intron key)
    ir_df['intron_key'] = ir_df.apply(
        lambda r: f"{r['chrom']}:{r['intron_start']}-{r['intron_end']}", axis=1
    )
    strand_map = dict(zip(ir_df['intron_key'], ir_df['strand']))

    # IR-only introns (not in any co-retained pair)
    ir_retained = ir_df[ir_df['ir_ratio'] >= args.min_ir_ratio]
    ir_only_coords = set(ir_retained['intron_key']) - coret_intron_coords

    print(f"  Co-retained introns: {len(coret_intron_coords)}")
    print(f"  IR-only introns:     {len(ir_only_coords)}")

    fasta = pysam.FastaFile(args.genome)

    def collect_sequences(coords: set, region: str) -> list[str]:
        seqs = []
        for coord in coords:
            strand = strand_map.get(coord, '+')
            windows = extract_boundary_sequences(coord, strand, fasta)
            if windows and region in windows:
                seqs.append(windows[region])
        return seqs

    regions = ['5p_intron', '3p_intron', '5p_exon', '3p_exon']
    all_results = []

    for region in regions:
        print(f"\nAnalysing {region}...")
        coret_seqs = collect_sequences(coret_intron_coords, region)
        ctrl_seqs = collect_sequences(ir_only_coords, region)
        print(f"  {len(coret_seqs)} co-retained seqs, {len(ctrl_seqs)} ctrl seqs")
        if len(coret_seqs) < 5 or len(ctrl_seqs) < 5:
            print("  Too few sequences — skipping")
            continue

        enrich = enrichment_test(coret_seqs, ctrl_seqs)
        enrich['region'] = region
        all_results.append(enrich)

        n_sig = (enrich['fdr'] < 0.05).sum()
        print(f"  Significant k-mers (FDR<0.05): {n_sig}")
        if n_sig > 0:
            top = enrich[enrich['fdr'] < 0.05].head(5)
            for _, r in top.iterrows():
                rbp = f" [{r['rbp_match']}]" if r['rbp_match'] else ''
                print(f"    {r['kmer']:8s} FC={r['fold_change']:.2f}  FDR={r['fdr']:.3e}{rbp}")

        plot_enrichment(enrich, args.outdir, sample_name, region)

    fasta.close()

    if all_results:
        out_df = pd.concat(all_results, ignore_index=True)
        out_tsv = os.path.join(args.outdir, f'{sample_name}_rbp_motifs.tsv')
        out_df.to_csv(out_tsv, sep='\t', index=False)
        print(f"\nSaved: {out_tsv}")

        # Summary heatmap: top k-mers × regions
        sig_all = out_df[out_df['fdr'] < 0.05]
        if len(sig_all) > 0:
            top_kmers = sig_all.groupby('kmer')['pvalue'].min().nsmallest(20).index
            pivot = sig_all[sig_all['kmer'].isin(top_kmers)].pivot_table(
                index='kmer', columns='region', values='fold_change', aggfunc='mean'
            ).fillna(1.0)
            fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.4)))
            sns.heatmap(np.log2(pivot), ax=ax, cmap='RdBu_r', center=0,
                        annot=True, fmt='.1f', linewidths=0.5)
            ax.set_title(f'Top enriched k-mers (log2 FC)\n{sample_name}')
            ax.set_xlabel('Sequence region')
            ax.set_ylabel('k-mer')
            plt.tight_layout()
            out_hm = os.path.join(args.outdir, 'figures', f'{sample_name}_rbp_heatmap.png')
            fig.savefig(out_hm, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {out_hm}")


if __name__ == "__main__":
    main()

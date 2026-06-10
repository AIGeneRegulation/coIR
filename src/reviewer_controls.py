"""
reviewer_controls.py — Two analyses addressing reviewer concerns.

Analysis 1: Expression-matched DepMap comparison.
  Compute per-gene expression (total reads across all introns) as a proxy.
  Bin all genes into expression quartiles. Within each quartile compare
  DepMap Chronos scores for co-retained vs IR-only genes (Mann-Whitney)
  and CORUM complex membership (Fisher's exact).
  Output: results/mechanistic/expression_matched/

Analysis 2: Transcript maturity check.
  For reads spanning significant co-retention pairs where BOTH focal introns
  are retained, count what fraction of OTHER introns on the same read are
  spliced. Compare to reads from non-significant pairs (both retained).
  Uses one well-powered sample per cell line (most significant pairs).
  Output: results/mechanistic/maturity_check/
"""

import os
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(__file__))
from ir_utils import _classify_read_at_intron, _make_chrom_map

# ── Sample → cell line mapping ─────────────────────────────────────────────────

SAMPLE_CELL: dict[str, str] = {
    'SGNex_A549_directRNA_replicate1_run1': 'A549',
    'SGNex_A549_directRNA_replicate4_run1': 'A549',
    'SGNex_A549_directRNA_replicate5_run1': 'A549',
    'SGNex_A549_directRNA_replicate6_run1': 'A549',
    'SGNex_HEYA8_directRNA_replicate1_run1': 'HEYA8',
    'SGNex_HEYA8_directRNA_replicate1_run2': 'HEYA8',
    'SGNex_HEYA8_directRNA_replicate2_run1': 'HEYA8',
    'SGNex_HEYA8_directRNA_replicate2_run2': 'HEYA8',
    'SGNex_HEYA8_directRNA_replicate3_run1': 'HEYA8',
    'SGNex_Hct116_directRNA_replicate1_run2': 'Hct116',
    'SGNex_Hct116_directRNA_replicate1_run3': 'Hct116',
    'SGNex_Hct116_directRNA_replicate2_run3': 'Hct116',
    'SGNex_Hct116_directRNA_replicate2_run6': 'Hct116',
    'SGNex_Hct116_directRNA_replicate3_run1': 'Hct116',
    'SGNex_Hct116_directRNA_replicate3_run4': 'Hct116',
    'SGNex_Hct116_directRNA_replicate4_run3': 'Hct116',
    'SGNex_HepG2_directRNA_replicate1_run3': 'HepG2',
    'SGNex_HepG2_directRNA_replicate5_run1': 'HepG2',
    'SGNex_HepG2_directRNA_replicate5_run2': 'HepG2',
    'SGNex_HepG2_directRNA_replicate6_run1': 'HepG2',
    'SGNex_K562_directRNA_replicate1_run1': 'K562',
    'SGNex_K562_directRNA_replicate4_run1': 'K562',
    'SGNex_K562_directRNA_replicate5_run1': 'K562',
    'SGNex_K562_directRNA_replicate6_run1': 'K562',
    'SGNex_MCF7-EV_directRNA_replicate1_run1': 'MCF7-EV',
    'SGNex_MCF7-EV_directRNA_replicate2_run1': 'MCF7-EV',
    'SGNex_MCF7_directRNA_replicate2_run2': 'MCF7',
    'SGNex_MCF7_directRNA_replicate2_run3': 'MCF7',
    'SGNex_MCF7_directRNA_replicate3_run1': 'MCF7',
    'SGNex_MCF7_directRNA_replicate4_run1': 'MCF7',
}

CLASS_COLORS = {'co_retained': 'coral', 'ir_only': 'steelblue'}
CLASS_LABELS = {'co_retained': 'Co-retained', 'ir_only': 'IR-only'}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def parse_coord(coord: str) -> tuple[str, int, int]:
    """'chr17:64500882-64500986' → (chrom, start, end)."""
    chrom, rest = coord.rsplit(':', 1)
    s, e = rest.split('-')
    return chrom, int(s), int(e)


def build_gene_classes(
    results_dir: str,
    fdr_threshold: float = 0.05,
    ir_threshold: float = 0.05,
) -> tuple[set, set, set, set]:
    """Classify genes as co_retained, ir_only, non_ir across all samples."""
    co_genes: set = set()
    all_ir: set = set()
    all_expressed: set = set()

    for sample in SAMPLE_CELL:
        coret_f = os.path.join(results_dir, sample, f'{sample}_coretention.tsv')
        ir_f    = os.path.join(results_dir, sample, f'{sample}_ir_events.tsv')

        if os.path.isfile(coret_f):
            try:
                df = pd.read_csv(coret_f, sep='\t')
                sig = df[df['fdr'] < fdr_threshold]
                co_genes.update(sig['gene_name'].dropna().unique())
            except Exception as exc:
                print(f'  WARN coretention {sample}: {exc}')

        if os.path.isfile(ir_f):
            try:
                df = pd.read_csv(ir_f, sep='\t')
                all_expressed.update(df['gene_name'].dropna().unique())
                all_ir.update(
                    df.loc[df['ir_ratio'] >= ir_threshold, 'gene_name'].dropna().unique()
                )
            except Exception as exc:
                print(f'  WARN ir_events {sample}: {exc}')

    ir_only = all_ir - co_genes
    non_ir  = all_expressed - all_ir - co_genes
    print(f'  co_retained:  {len(co_genes):>5} genes')
    print(f'  ir_only:      {len(ir_only):>5} genes')
    print(f'  non_ir:       {len(non_ir):>5} genes')
    return co_genes, ir_only, non_ir, all_expressed


def build_gene_expression(results_dir: str) -> dict[str, float]:
    """
    Compute per-gene expression proxy: median across samples of
    sum(splice_reads + intronic_reads) across all introns in that gene.
    """
    gene_sample_reads: dict[str, list] = defaultdict(list)
    for sample in SAMPLE_CELL:
        ir_f = os.path.join(results_dir, sample, f'{sample}_ir_events.tsv')
        if not os.path.isfile(ir_f):
            continue
        try:
            df = pd.read_csv(ir_f, sep='\t')
            df['total'] = df['splice_reads'] + df['intronic_reads']
            for gene, total in df.groupby('gene_name')['total'].sum().items():
                gene_sample_reads[gene].append(float(total))
        except Exception as exc:
            print(f'  WARN expression {sample}: {exc}')
    return {gene: float(np.median(vals)) for gene, vals in gene_sample_reads.items()}


def load_depmap(data_dir: str) -> dict[str, float] | None:
    """Load DepMap Chronos scores; return {gene_symbol: median_across_cell_lines}."""
    candidates = [
        os.path.join(data_dir, 'depmap', 'CRISPRGeneEffect.csv'),
        os.path.join(data_dir, 'depmap', 'Chronos_Combined.csv'),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        print(f'  [SKIP] DepMap not found in {data_dir}/depmap/')
        return None
    print(f'  Loading DepMap: {path}')
    df = pd.read_csv(path, index_col=0)
    col_map = {col: col.split(' (')[0].strip() if ' (' in col else col.strip()
               for col in df.columns}
    df.rename(columns=col_map, inplace=True)
    medians = df.median(axis=0)
    return dict(zip(medians.index, medians.values))


def load_corum(data_dir: str) -> set | None:
    """Return set of gene symbols appearing in any human CORUM complex."""
    import zipfile
    candidates = [
        os.path.join(data_dir, 'corum', 'allComplexes.txt'),
        os.path.join(data_dir, 'corum', 'humanComplexes.txt'),
        os.path.join(data_dir, 'corum', 'allComplexes.txt.zip'),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        print(f'  [SKIP] CORUM not found in {data_dir}/corum/')
        return None
    print(f'  Loading CORUM: {path}')
    if path.endswith('.zip'):
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                inner = [n for n in zf.namelist() if n.endswith('.txt')]
                if not inner:
                    return None
                with zf.open(inner[0]) as fh:
                    corum_df = pd.read_csv(fh, sep='\t')
        except zipfile.BadZipFile:
            print('  [SKIP] CORUM zip corrupt')
            return None
    else:
        corum_df = pd.read_csv(path, sep='\t')

    org_col = next((c for c in corum_df.columns
                    if c.lower() in ('organism', 'organism_ncbi_id')), None)
    if org_col:
        corum_df = corum_df[
            corum_df[org_col].astype(str).str.contains('Human|9606', case=False, na=False)
        ]
    gene_col = next((c for c in corum_df.columns
                     if 'gene name' in c.lower() or 'subunit' in c.lower()), None)
    if gene_col is None:
        print('  [SKIP] CORUM: cannot find gene column')
        return None
    genes: set = set()
    for raw in corum_df[gene_col].dropna():
        genes.update(g.strip() for g in str(raw).split(';') if g.strip())
    print(f'  {len(genes)} genes in CORUM complexes')
    return genes


# ── Analysis 1: Expression-matched DepMap comparison ──────────────────────────

def analysis_expression_matched(
    results_dir: str,
    data_dir: str,
    outdir: str,
    fdr_threshold: float = 0.05,
    ir_threshold: float = 0.05,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)

    print('\n[1] Expression-matched DepMap comparison')
    print('=' * 60)

    print('Building gene classes...')
    co_genes, ir_only, non_ir, all_expressed = build_gene_classes(
        results_dir, fdr_threshold, ir_threshold)

    print('Building gene expression estimates...')
    expr = build_gene_expression(results_dir)
    print(f'  Expression estimates for {len(expr)} genes')

    chronos = load_depmap(data_dir)
    if chronos is None:
        return

    corum_genes = load_corum(data_dir)

    # Build gene-level DataFrame (only co_retained and ir_only for comparison)
    rows = []
    for gene in co_genes | ir_only:
        if gene not in expr or gene not in chronos:
            continue
        rows.append({
            'gene':       gene,
            'gene_class': 'co_retained' if gene in co_genes else 'ir_only',
            'expression': expr[gene],
            'chronos':    chronos[gene],
            'in_corum':   int(gene in corum_genes) if corum_genes is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    print(f'  {len(df)} genes with expression + Chronos data'
          f' ({(df.gene_class == "co_retained").sum()} co_retained,'
          f' {(df.gene_class == "ir_only").sum()} ir_only)')

    if len(df) < 8:
        print('  Too few genes for quartile analysis — exiting')
        return

    # Quartile assignment on expression (across both classes together)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df['expr_quartile'] = pd.qcut(df['expression'], q=4,
                                       labels=['Q1', 'Q2', 'Q3', 'Q4'])

    df.to_csv(os.path.join(outdir, 'gene_expression_class_chronos.tsv'), sep='\t', index=False)

    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
    q_labels  = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']

    # Per-quartile statistics
    chronos_rows = []
    corum_rows   = []

    for q in quartiles:
        sub = df[df['expr_quartile'] == q]
        co_ch = sub.loc[sub['gene_class'] == 'co_retained', 'chronos'].dropna().values
        ir_ch = sub.loc[sub['gene_class'] == 'ir_only',     'chronos'].dropna().values

        row: dict = {
            'quartile':         q,
            'n_co_retained':    len(co_ch),
            'n_ir_only':        len(ir_ch),
            'median_co':        float(np.median(co_ch)) if len(co_ch) else np.nan,
            'median_ir':        float(np.median(ir_ch)) if len(ir_ch) else np.nan,
            'mw_stat':          np.nan,
            'mw_pvalue':        np.nan,
        }
        if len(co_ch) >= 3 and len(ir_ch) >= 3:
            stat, pval = mannwhitneyu(co_ch, ir_ch, alternative='less')
            row['mw_stat']   = float(stat)
            row['mw_pvalue'] = float(pval)
        chronos_rows.append(row)

        # CORUM comparison within quartile
        if corum_genes is not None:
            co_q = set(sub.loc[sub['gene_class'] == 'co_retained', 'gene'])
            ir_q = set(sub.loc[sub['gene_class'] == 'ir_only',     'gene'])
            co_in  = len(co_q & corum_genes)
            co_out = len(co_q) - co_in
            ir_in  = len(ir_q & corum_genes)
            ir_out = len(ir_q) - ir_in
            if co_in + co_out > 0 and ir_in + ir_out > 0:
                or_, pval_c = fisher_exact([[co_in, co_out], [ir_in, ir_out]],
                                           alternative='greater')
            else:
                or_, pval_c = np.nan, np.nan
            corum_rows.append({
                'quartile':      q,
                'co_in_corum':   co_in,
                'co_total':      len(co_q),
                'ir_in_corum':   ir_in,
                'ir_total':      len(ir_q),
                'pct_co':        co_in / len(co_q) * 100 if co_q else 0.0,
                'pct_ir':        ir_in / len(ir_q) * 100 if ir_q else 0.0,
                'fisher_or':     float(or_),
                'fisher_pvalue': float(pval_c),
            })

    # FDR-correct Chronos tests across quartiles
    chron_df = pd.DataFrame(chronos_rows)
    pvals    = chron_df['mw_pvalue'].fillna(1.0).tolist()
    if any(not np.isnan(p) for p in pvals):
        _, fdr, _, _ = multipletests(pvals, method='fdr_bh')
        chron_df['fdr'] = fdr
    chron_df.to_csv(os.path.join(outdir, 'chronos_by_expression_quartile.tsv'),
                    sep='\t', index=False)
    print('\n  Chronos results by quartile:')
    print(chron_df[['quartile', 'n_co_retained', 'n_ir_only',
                     'median_co', 'median_ir', 'mw_pvalue', 'fdr']].to_string(index=False))

    if corum_rows:
        corum_df = pd.DataFrame(corum_rows)
        corum_df.to_csv(os.path.join(outdir, 'corum_by_expression_quartile.tsv'),
                        sep='\t', index=False)
        print('\n  CORUM membership by quartile:')
        print(corum_df[['quartile', 'pct_co', 'pct_ir',
                         'fisher_or', 'fisher_pvalue']].to_string(index=False))

    # ── Figure: side-by-side boxplots per quartile ────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for i, (q, ql) in enumerate(zip(quartiles, q_labels)):
        ax   = axes[i]
        sub  = df[df['expr_quartile'] == q]
        co_d = sub.loc[sub['gene_class'] == 'co_retained', 'chronos'].dropna().values
        ir_d = sub.loc[sub['gene_class'] == 'ir_only',     'chronos'].dropna().values

        bp = ax.boxplot([co_d, ir_d], patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', lw=1.5),
                        widths=0.5)
        for patch, color in zip(bp['boxes'], ['coral', 'steelblue']):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Co-retained', 'IR-only'], fontsize=8, rotation=20, ha='right')
        ax.set_title(ql, fontsize=9)
        ax.axhline(0, lw=0.5, color='grey', linestyle='--')

        # Sample sizes below boxes
        ax.text(1, ax.get_ylim()[0], f'n={len(co_d)}',
                ha='center', va='top', fontsize=7, color='grey')
        ax.text(2, ax.get_ylim()[0], f'n={len(ir_d)}',
                ha='center', va='top', fontsize=7, color='grey')

        # p-value annotation
        qrow = chron_df[chron_df['quartile'] == q].iloc[0]
        pval = qrow['mw_pvalue']
        pstr = f'p={pval:.2e}' if not np.isnan(pval) else 'n.d.'
        ax.set_xlabel(pstr, fontsize=7, color='dimgrey')

        if i == 0:
            ax.set_ylabel('Median Chronos score\n(more negative = more essential)', fontsize=9)

    fig.suptitle(
        'DepMap Chronos scores by expression quartile: Co-retained vs IR-only',
        fontsize=11, y=1.01,
    )
    figpath = os.path.join(outdir, 'figures', 'chronos_by_expression_quartile.png')
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Figure → {figpath}')
    print(f'  TSVs → {outdir}/')


# ── Analysis 2: Transcript maturity check ─────────────────────────────────────

def find_best_sample_per_cell_line(
    results_dir: str,
    bam_dir: str,
    fdr_threshold: float = 0.05,
) -> dict[str, str]:
    """
    Scan all sample coretention TSVs; for each cell line pick the sample
    with the most significant co-retention pairs that also has a BAM file.
    """
    cell_best: dict[str, tuple[int, str]] = {}  # cell → (n_sig, sample)

    all_samples = [
        d for d in os.listdir(results_dir)
        if d.startswith('SGNex_') and os.path.isdir(os.path.join(results_dir, d))
    ]

    for sample in all_samples:
        bam = os.path.join(bam_dir, f'{sample}.bam')
        if not os.path.isfile(bam):
            continue
        coret_f = os.path.join(results_dir, sample, f'{sample}_coretention.tsv')
        if not os.path.isfile(coret_f):
            continue

        # Derive cell line name from sample name
        parts = sample.split('_')
        # Format: SGNex_CELLLINE_directRNA_replicateN_runM
        cell = parts[1]  # e.g. A549, HEYA8, Hct116, HepG2, K562, MCF7-EV, MCF7

        try:
            df    = pd.read_csv(coret_f, sep='\t')
            n_sig = int((df['fdr'] < fdr_threshold).sum())
        except Exception:
            continue

        prev_best = cell_best.get(cell, (-1, ''))
        if n_sig > prev_best[0]:
            cell_best[cell] = (n_sig, sample)

    result = {}
    for cell, (n_sig, sample) in cell_best.items():
        result[cell] = sample
        print(f'  {cell:<10}: {sample}  ({n_sig} sig pairs)')
    return result


def build_gene_introns_from_ir(ir_df: pd.DataFrame) -> dict[str, list]:
    """
    Build {gene_id: [(chrom, intron_start, intron_end), ...]} from ir_events DataFrame.
    Deduplicates introns (same gene can have multiple transcripts).
    """
    gene_introns: dict[str, set] = defaultdict(set)
    for _, row in ir_df.iterrows():
        gene_introns[row['gene_id']].add(
            (row['chrom'], int(row['intron_start']), int(row['intron_end']))
        )
    return {gid: sorted(introns) for gid, introns in gene_introns.items()}


def classify_other_introns(
    read: pysam.AlignedSegment,
    gene_id: str,
    focal_a: tuple[str, int, int],
    focal_b: tuple[str, int, int],
    gene_introns: dict[str, list],
    chrom_map: dict,
    min_overhang: int = 10,
) -> tuple[int, int]:
    """
    For a read spanning focal introns a and b, classify all OTHER gene introns
    the read covers. Returns (n_spliced, n_retained) for non-focal introns.
    Introns that span outside the read or return 'ambiguous' are skipped.
    """
    n_spliced  = 0
    n_retained = 0
    focal_set  = {(focal_a[1], focal_a[2]), (focal_b[1], focal_b[2])}

    for (chrom, istart, iend) in gene_introns.get(gene_id, []):
        if (istart, iend) in focal_set:
            continue
        # Read must span this intron with at least min_overhang on each side
        if read.reference_start > istart - min_overhang:
            continue
        if read.reference_end is None or read.reference_end < iend + min_overhang:
            continue
        state = _classify_read_at_intron(read, istart, iend)
        if state == 'spliced':
            n_spliced += 1
        elif state == 'retained':
            n_retained += 1
    return n_spliced, n_retained


def maturity_check_sample(
    sample: str,
    bam_path: str,
    coret_df: pd.DataFrame,
    ir_df: pd.DataFrame,
    outdir: str,
    fdr_threshold: float = 0.05,
    max_sig_pairs: int = 300,
    max_nonsig_pairs: int = 300,
    max_reads_per_pair: int = 60,
    min_reads_filter: int = 10,
) -> pd.DataFrame | None:
    """
    Compute transcript maturity metric for one sample.
    Returns per-read DataFrame with columns: group, n_spliced, n_retained, frac_spliced.
    """
    gene_introns = build_gene_introns_from_ir(ir_df)

    # Build gene_id lookup from coretention (gene_name → gene_id mapping from ir_events)
    gene_name_to_id: dict[str, str] = {}
    for _, row in ir_df[['gene_name', 'gene_id']].drop_duplicates().iterrows():
        gene_name_to_id[row['gene_name']] = row['gene_id']

    sig_df    = coret_df[coret_df['fdr'] < fdr_threshold].copy()
    nonsig_df = coret_df[
        (coret_df['fdr'] >= fdr_threshold) &
        (coret_df['total_spanning_reads'] >= min_reads_filter)
    ].copy()

    # Sample non-sig pairs (random)
    if len(nonsig_df) > max_nonsig_pairs:
        nonsig_df = nonsig_df.sample(max_nonsig_pairs, random_state=42)

    # Take top sig pairs by significance
    if len(sig_df) > max_sig_pairs:
        sig_df = sig_df.nsmallest(max_sig_pairs, 'fdr')

    print(f'  {sample}: {len(sig_df)} sig pairs, {len(nonsig_df)} non-sig pairs')

    bamfile  = pysam.AlignmentFile(bam_path, 'rb')
    # Build a flat list of all intron objects for _make_chrom_map
    all_intron_chroms = set(ir_df['chrom'].unique())
    # Quick chrom normalisation: detect chr prefix from BAM headers
    bam_refs = set(bamfile.references)
    use_chr  = any(r.startswith('chr') for r in list(bam_refs)[:5])

    def _norm_chrom(c: str) -> str:
        if use_chr and not c.startswith('chr'):
            return 'chr' + c
        if not use_chr and c.startswith('chr'):
            return c[3:]
        return c

    read_rows: list[dict] = []

    def _process_pairs(pair_df: pd.DataFrame, group: str) -> None:
        for _, row in pair_df.iterrows():
            try:
                chrom_a, start_a, end_a = parse_coord(row['intron_a'])
                chrom_b, start_b, end_b = parse_coord(row['intron_b'])
            except Exception:
                continue

            bam_chrom = _norm_chrom(chrom_a)
            if bam_chrom not in bam_refs:
                continue

            region_start = start_a - 10
            region_end   = end_b   + 10
            if region_end - region_start > 50_000:
                continue

            gene_name = row.get('gene_name', '')
            gene_id   = gene_name_to_id.get(gene_name, gene_name)

            both_ret_count = 0
            try:
                reads = bamfile.fetch(bam_chrom, max(0, region_start), region_end)
            except (ValueError, KeyError):
                continue

            for read in reads:
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                if read.reference_start > region_start:
                    continue
                if read.reference_end is None or read.reference_end < region_end:
                    continue

                state_a = _classify_read_at_intron(read, start_a, end_a)
                state_b = _classify_read_at_intron(read, start_b, end_b)

                # Only process reads where both focal introns are retained
                if state_a != 'retained' or state_b != 'retained':
                    continue

                n_spl, n_ret = classify_other_introns(
                    read,
                    gene_id,
                    (chrom_a, start_a, end_a),
                    (chrom_b, start_b, end_b),
                    gene_introns,
                    {},
                )
                n_other = n_spl + n_ret
                frac    = n_spl / n_other if n_other > 0 else np.nan

                read_rows.append({
                    'group':       group,
                    'sample':      sample,
                    'gene':        gene_name,
                    'intron_a':    row['intron_a'],
                    'intron_b':    row['intron_b'],
                    'n_spliced':   n_spl,
                    'n_retained':  n_ret,
                    'n_other':     n_other,
                    'frac_spliced': frac,
                })
                both_ret_count += 1
                if both_ret_count >= max_reads_per_pair:
                    break

    _process_pairs(sig_df,    'co_retained')
    _process_pairs(nonsig_df, 'non_sig')
    bamfile.close()

    if not read_rows:
        print(f'  No qualifying reads found for {sample}')
        return None

    df_out = pd.DataFrame(read_rows)
    out_tsv = os.path.join(outdir, f'{sample}_maturity.tsv')
    df_out.to_csv(out_tsv, sep='\t', index=False)
    print(f'  Saved {len(df_out)} read observations → {out_tsv}')
    return df_out


def analysis_maturity_check(
    results_dir: str,
    bam_dir: str,
    outdir: str,
    fdr_threshold: float = 0.05,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'figures'), exist_ok=True)

    print('\n[2] Transcript maturity check')
    print('=' * 60)

    print('Finding best sample per cell line...')
    best_samples = find_best_sample_per_cell_line(results_dir, bam_dir, fdr_threshold)

    if not best_samples:
        print('  No samples with BAMs found — skipping')
        return

    all_dfs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for cell, sample in sorted(best_samples.items()):
        print(f'\n  Processing {cell}: {sample}')
        bam_path = os.path.join(bam_dir, f'{sample}.bam')
        coret_f  = os.path.join(results_dir, sample, f'{sample}_coretention.tsv')
        ir_f     = os.path.join(results_dir, sample, f'{sample}_ir_events.tsv')

        if not all(os.path.isfile(p) for p in [bam_path, coret_f, ir_f]):
            print(f'  Missing files for {sample} — skipping')
            continue

        coret_df = pd.read_csv(coret_f, sep='\t')
        ir_df    = pd.read_csv(ir_f, sep='\t')

        df_sample = maturity_check_sample(
            sample, bam_path, coret_df, ir_df, outdir, fdr_threshold,
        )
        if df_sample is None:
            continue
        df_sample['cell_line'] = cell
        all_dfs.append(df_sample)

        # Per-sample summary stats
        for group in ('co_retained', 'non_sig'):
            sub = df_sample[df_sample['group'] == group]['frac_spliced'].dropna()
            summary_rows.append({
                'sample':     sample,
                'cell_line':  cell,
                'group':      group,
                'n_reads':    len(sub),
                'n_reads_with_other_introns': (df_sample[df_sample['group'] == group]['n_other'] > 0).sum(),
                'median_frac_spliced': float(np.median(sub)) if len(sub) else np.nan,
                'mean_frac_spliced':   float(np.mean(sub))   if len(sub) else np.nan,
            })

    if not all_dfs:
        print('  No maturity data collected — skipping plots')
        return

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv(os.path.join(outdir, 'maturity_all_reads.tsv'), sep='\t', index=False)

    summary_df = pd.DataFrame(summary_rows)

    # Aggregate test across all samples
    co_frac = all_df.loc[
        (all_df['group'] == 'co_retained') & all_df['frac_spliced'].notna(),
        'frac_spliced'
    ].values
    ns_frac = all_df.loc[
        (all_df['group'] == 'non_sig') & all_df['frac_spliced'].notna(),
        'frac_spliced'
    ].values

    stat, pval = (np.nan, np.nan)
    if len(co_frac) >= 3 and len(ns_frac) >= 3:
        stat, pval = mannwhitneyu(co_frac, ns_frac, alternative='two-sided')

    summary_df.to_csv(os.path.join(outdir, 'maturity_summary.tsv'), sep='\t', index=False)

    agg_row = {
        'n_co_retained_reads':   len(co_frac),
        'n_nonsig_reads':        len(ns_frac),
        'median_frac_spliced_co': float(np.median(co_frac)) if len(co_frac) else np.nan,
        'median_frac_spliced_ns': float(np.median(ns_frac)) if len(ns_frac) else np.nan,
        'mw_stat':               float(stat) if not np.isnan(stat) else np.nan,
        'mw_pvalue':             float(pval) if not np.isnan(pval) else np.nan,
    }
    pd.DataFrame([agg_row]).to_csv(
        os.path.join(outdir, 'maturity_aggregate_test.tsv'), sep='\t', index=False)

    print('\n  Aggregate maturity test:')
    print(f'    co_retained reads:  n={len(co_frac):>5}  '
          f'median frac spliced = {np.median(co_frac):.3f}' if len(co_frac) else '    co_retained: 0 reads')
    print(f'    non_sig reads:      n={len(ns_frac):>5}  '
          f'median frac spliced = {np.median(ns_frac):.3f}' if len(ns_frac) else '    non_sig: 0 reads')
    print(f'    Mann-Whitney U: stat={stat:.1f}  p={pval:.3e}' if not np.isnan(stat) else '    MWU: insufficient data')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: violin/box of frac_spliced, co_retained vs non_sig
    ax = axes[0]
    co_vals = all_df.loc[(all_df['group'] == 'co_retained') & all_df['frac_spliced'].notna(),
                         'frac_spliced'].values
    ns_vals = all_df.loc[(all_df['group'] == 'non_sig') & all_df['frac_spliced'].notna(),
                         'frac_spliced'].values
    bp = ax.boxplot([co_vals, ns_vals], patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=1.5), widths=0.5)
    bp['boxes'][0].set_facecolor('coral');    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][1].set_facecolor('lightgrey'); bp['boxes'][1].set_alpha(0.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Co-retained\n(sig pairs)', 'Non-significant\npairs'], fontsize=9)
    ax.set_ylabel('Fraction of other introns spliced', fontsize=9)
    ax.set_title('Transcript maturity: other introns\n(reads with both focal introns retained)', fontsize=9)
    pstr = f'MWU p={pval:.2e}' if not np.isnan(pval) else ''
    ax.set_xlabel(pstr, fontsize=8, color='dimgrey')
    ax.text(1, ax.get_ylim()[0], f'n={len(co_vals)}', ha='center', va='top', fontsize=7, color='grey')
    ax.text(2, ax.get_ylim()[0], f'n={len(ns_vals)}', ha='center', va='top', fontsize=7, color='grey')
    ax.axhline(1.0, lw=0.5, color='grey', linestyle='--')

    # Panel 2: per-cell-line median frac_spliced
    ax2 = axes[1]
    cell_lines = sorted(all_df['cell_line'].unique())
    x = np.arange(len(cell_lines))
    w = 0.35
    for j, (group, color, label) in enumerate([
        ('co_retained', 'coral',     'Co-retained'),
        ('non_sig',     'lightgrey', 'Non-significant'),
    ]):
        medians = []
        for cl in cell_lines:
            sub = all_df.loc[(all_df['cell_line'] == cl) & (all_df['group'] == group),
                             'frac_spliced'].dropna()
            medians.append(float(np.median(sub)) if len(sub) else np.nan)
        ax2.bar(x + (j - 0.5) * w, medians, w, label=label, color=color, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cell_lines, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Median fraction of other introns spliced', fontsize=9)
    ax2.set_title('Maturity by cell line', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.axhline(1.0, lw=0.5, color='grey', linestyle='--')

    fig.suptitle('Transcript Maturity Check: Co-retention vs Background', fontsize=11)
    plt.tight_layout()
    figpath = os.path.join(outdir, 'figures', 'maturity_check.png')
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Figure → {figpath}')
    print(f'  TSVs → {outdir}/')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir',      default='results')
    parser.add_argument('--data-dir',         default='data')
    parser.add_argument('--bam-dir',          default='data/raw/sgnex')
    parser.add_argument('--outdir-expr',      default='results/mechanistic/expression_matched')
    parser.add_argument('--outdir-maturity',  default='results/mechanistic/maturity_check')
    parser.add_argument('--fdr-threshold',    type=float, default=0.05)
    parser.add_argument('--ir-threshold',     type=float, default=0.05)
    args = parser.parse_args()

    print('=' * 60)
    print('Reviewer controls analysis')
    print(f'  results_dir     : {args.results_dir}')
    print(f'  data_dir        : {args.data_dir}')
    print(f'  bam_dir         : {args.bam_dir}')
    print(f'  outdir_expr     : {args.outdir_expr}')
    print(f'  outdir_maturity : {args.outdir_maturity}')
    print('=' * 60)

    analysis_expression_matched(
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        outdir=args.outdir_expr,
        fdr_threshold=args.fdr_threshold,
        ir_threshold=args.ir_threshold,
    )

    analysis_maturity_check(
        results_dir=args.results_dir,
        bam_dir=args.bam_dir,
        outdir=args.outdir_maturity,
        fdr_threshold=args.fdr_threshold,
    )

    print('\n' + '=' * 60)
    print('All done.')
    print('=' * 60)


if __name__ == '__main__':
    main()

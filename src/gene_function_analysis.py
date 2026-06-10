"""
gene_function_analysis.py — Five gene-function analyses on co-retained genes.

Reads per-sample coretention/ir_events TSVs from results/SGNex_*/ and external
database files from data/ (GO ontology, gnomAD, DepMap, CORUM — see README for
download instructions if any files are missing).

Analyses
--------
1. GO enrichment (BP/MF/CC): hypergeometric + BH correction, top 20 terms
2. gnomAD dosage sensitivity: pLI and LOEUF across co-retained / IR-only / non-IR
3. DepMap essentiality: Chronos scores across gene classes
4. Constitutive vs cell-type-specific co-retention + GO enrichment per category
5. CORUM protein complex membership enrichment vs background

Output: results/mechanistic/gene_function/
"""

import argparse
import glob
import gzip
import os
import zipfile
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, hypergeom, mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ── Constants ──────────────────────────────────────────────────────────────────

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

NS_SHORT = {
    'biological_process': 'BP',
    'molecular_function': 'MF',
    'cellular_component': 'CC',
}

CLASS_COLORS = {
    'co_retained': 'coral',
    'ir_only':     'steelblue',
    'non_ir':      'lightgrey',
}

CLASS_LABELS = {
    'co_retained': 'Co-retained',
    'ir_only':     'IR-only',
    'non_ir':      'Non-IR',
}


# ── Gene class construction ────────────────────────────────────────────────────

def build_gene_classes(
    results_dir: str,
    fdr_threshold: float = 0.05,
    ir_threshold: float = 0.05,
) -> tuple[set, set, set, set, dict]:
    """
    Scan all per-sample coretention and ir_events TSVs to classify genes.

    Returns
    -------
    co_retained_genes  : genes with ≥1 significant co-retention pair in any sample
    ir_only_genes      : genes with IR (ir_ratio ≥ threshold) but no co-retention
    non_ir_genes       : expressed genes with neither IR nor co-retention
    all_expressed      : union of all expressed genes
    gene_to_cell_lines : {gene: set of cell lines with significant co-retention}
    """
    co_genes: set = set()
    all_ir: set = set()
    all_expressed: set = set()
    gene_to_cl: dict = defaultdict(set)

    for sample, cell in SAMPLE_CELL.items():
        coret_f = os.path.join(results_dir, sample, f'{sample}_coretention.tsv')
        ir_f    = os.path.join(results_dir, sample, f'{sample}_ir_events.tsv')

        if os.path.isfile(coret_f):
            try:
                df = pd.read_csv(coret_f, sep='\t')
                sig = df[df['fdr'] < fdr_threshold]
                for g in sig['gene_name'].unique():
                    co_genes.add(g)
                    gene_to_cl[g].add(cell)
            except Exception as e:
                print(f'  WARN coretention {sample}: {e}')

        if os.path.isfile(ir_f):
            try:
                df = pd.read_csv(ir_f, sep='\t')
                all_expressed.update(df['gene_name'].dropna().unique())
                all_ir.update(
                    df.loc[df['ir_ratio'] >= ir_threshold, 'gene_name'].dropna().unique()
                )
            except Exception as e:
                print(f'  WARN ir_events {sample}: {e}')

    ir_only  = all_ir - co_genes
    non_ir   = all_expressed - all_ir - co_genes
    print(f'  co_retained : {len(co_genes):>5} genes')
    print(f'  ir_only     : {len(ir_only):>5} genes')
    print(f'  non_ir      : {len(non_ir):>5} genes')
    print(f'  all_expressed: {len(all_expressed):>5} genes')
    return co_genes, ir_only, non_ir, all_expressed, dict(gene_to_cl)


# ── GO parsing ────────────────────────────────────────────────────────────────

def parse_obo(obo_path: str) -> tuple[dict, dict]:
    """
    Parse go-basic.obo.

    Returns
    -------
    terms   : {go_id: {'name': str, 'namespace': str}}
    ancestors: {go_id: frozenset of all ancestor go_ids (excluding self)}
    """
    terms: dict = {}
    parents: dict = defaultdict(set)  # direct parents only

    cur_id = cur_name = cur_ns = None
    obsolete = False

    def _flush():
        nonlocal cur_id, cur_name, cur_ns, obsolete
        if cur_id and cur_name and cur_ns and not obsolete:
            terms[cur_id] = {'name': cur_name, 'namespace': cur_ns}
        cur_id = cur_name = cur_ns = None
        obsolete = False

    open_fn = gzip.open if obo_path.endswith('.gz') else open
    with open_fn(obo_path, 'rt', encoding='utf-8', errors='replace') as fh:
        in_term = False
        for line in fh:
            line = line.strip()
            if line == '[Term]':
                _flush()
                in_term = True
            elif line.startswith('[') and line != '[Term]':
                _flush()
                in_term = False
            elif not in_term:
                continue
            elif line.startswith('id: '):
                cur_id = line[4:]
            elif line.startswith('name: '):
                cur_name = line[6:]
            elif line.startswith('namespace: '):
                cur_ns = line[11:]
            elif line.startswith('is_obsolete: true'):
                obsolete = True
            elif line.startswith('is_a: ') or line.startswith('part_of: '):
                raw = line.split(': ', 1)[1].split(' ! ')[0].strip()
                if cur_id:
                    parents[cur_id].add(raw)
        _flush()

    # BFS to compute all ancestors for every term
    ancestors: dict = {}
    for tid in terms:
        anc: set = set()
        queue = list(parents.get(tid, []))
        while queue:
            p = queue.pop()
            if p not in anc:
                anc.add(p)
                queue.extend(parents.get(p, []))
        ancestors[tid] = frozenset(anc)

    print(f'  Parsed {len(terms)} GO terms')
    return terms, ancestors


def load_gaf(
    gaf_path: str,
    gene_universe: set | None = None,
) -> dict:
    """
    Parse goa_human.gaf or goa_human.gaf.gz.

    Returns {gene_symbol: set of direct GO IDs (before ancestor propagation)}.
    Skips NOT qualifiers and genes outside gene_universe (if provided).
    """
    gene2go: dict = defaultdict(set)
    open_fn = gzip.open if gaf_path.endswith('.gz') else open
    n_lines = 0
    with open_fn(gaf_path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('!'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue
            qualifier = parts[3]
            if 'NOT' in qualifier:
                continue
            symbol = parts[2]
            go_id  = parts[4]
            if gene_universe and symbol not in gene_universe:
                continue
            gene2go[symbol].add(go_id)
            n_lines += 1
    print(f'  Parsed {n_lines} GAF annotations for {len(gene2go)} genes')
    return dict(gene2go)


def propagate_annotations(gene2go: dict, ancestors: dict) -> dict:
    """Add all ancestor GO IDs to each gene's annotation set."""
    propagated: dict = {}
    for gene, goids in gene2go.items():
        expanded: set = set(goids)
        for goid in goids:
            expanded.update(ancestors.get(goid, frozenset()))
        propagated[gene] = expanded
    return propagated


# ── GO enrichment ─────────────────────────────────────────────────────────────

def run_go_enrichment(
    query_genes: set,
    background_genes: set,
    gene2go_prop: dict,
    terms: dict,
    namespace: str,
    min_term_size: int = 5,
    max_term_size: int = 500,
) -> pd.DataFrame:
    """
    Hypergeometric over-representation test for one GO namespace.

    M = |background ∩ annotated|  (genes with ≥1 annotation in this ns)
    N = |query ∩ background|
    n = |background ∩ annotated_to_T|
    k = |query ∩ annotated_to_T|
    """
    bg_genes = background_genes
    qr_genes = query_genes & bg_genes

    # Restrict to genes with any annotation in this namespace
    ns_terms = {tid for tid, info in terms.items() if info['namespace'] == namespace}

    # Build term → gene sets over background
    term_bg: dict = defaultdict(set)
    for gene in bg_genes:
        for goid in gene2go_prop.get(gene, set()):
            if goid in ns_terms:
                term_bg[goid].add(gene)

    M = len(bg_genes)
    N = len(qr_genes)
    rows = []

    for tid, bg_set in term_bg.items():
        n = len(bg_set)
        if n < min_term_size or n > max_term_size:
            continue
        k = len(qr_genes & bg_set)
        if k == 0:
            continue
        pval = float(hypergeom.sf(k - 1, M, n, N))
        fold = (k / N) / (n / M) if N > 0 and M > 0 else float('nan')
        rows.append({
            'go_id':     tid,
            'go_name':   terms[tid]['name'],
            'namespace': namespace,
            'k':         k,
            'n_bg':      n,
            'N_query':   N,
            'M_bg':      M,
            'fold_enrichment': fold,
            'pvalue':    pval,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    _, fdr, _, _ = multipletests(df['pvalue'].values, method='fdr_bh')
    df['fdr'] = fdr
    df.sort_values(['fdr', 'pvalue'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def plot_go_dotplot(
    df: pd.DataFrame,
    title: str,
    outpath: str,
    top_n: int = 20,
) -> None:
    """Dot plot: x = fold enrichment, y = term name, size = k, color = -log10 FDR."""
    sub = df[df['fdr'] < 0.05].head(top_n)
    if sub.empty:
        print(f'  No significant terms for {title}')
        return

    sub = sub.copy()
    sub['log10fdr'] = -np.log10(sub['fdr'].clip(lower=1e-300))
    sub['go_label'] = sub.apply(
        lambda r: f"{r['go_id']} {r['go_name'][:55]}", axis=1
    )
    sub = sub.sort_values('fold_enrichment', ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(sub) * 0.35 + 1)))
    sc = ax.scatter(
        sub['fold_enrichment'], sub['go_label'],
        s=np.clip(sub['k'] * 3, 10, 200),
        c=sub['log10fdr'],
        cmap='YlOrRd',
        vmin=0,
        alpha=0.85,
        edgecolors='grey',
        linewidths=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label('−log₁₀ FDR', fontsize=8)
    ax.set_xlabel('Fold enrichment', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', labelsize=8)
    ax.axvline(1.0, lw=0.5, color='grey', linestyle='--')
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure → {outpath}')


# ── Analysis 1: GO enrichment ─────────────────────────────────────────────────

def analysis_go(
    co_genes: set,
    all_expressed: set,
    data_dir: str,
    outdir: str,
) -> None:
    obo_path = os.path.join(data_dir, 'go', 'go-basic.obo')
    gaf_paths = [
        os.path.join(data_dir, 'go', 'goa_human.gaf.gz'),
        os.path.join(data_dir, 'go', 'goa_human.gaf'),
    ]
    gaf_path = next((p for p in gaf_paths if os.path.isfile(p)), None)

    if not os.path.isfile(obo_path):
        print(f'  [SKIP] go-basic.obo not found at {obo_path}')
        return
    if gaf_path is None:
        print(f'  [SKIP] goa_human.gaf(.gz) not found in {data_dir}/go/')
        return

    print(f'  Parsing OBO: {obo_path}')
    terms, ancestors = parse_obo(obo_path)

    print(f'  Parsing GAF: {gaf_path}')
    gene2go_raw = load_gaf(gaf_path, gene_universe=all_expressed)

    print('  Propagating annotations...')
    gene2go_prop = propagate_annotations(gene2go_raw, ancestors)

    all_rows = []
    fig_dir = os.path.join(outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    for ns in ('biological_process', 'molecular_function', 'cellular_component'):
        ns_short = NS_SHORT[ns]
        print(f'  Running {ns_short} enrichment ({len(co_genes)} query, '
              f'{len(all_expressed)} background)...')
        df = run_go_enrichment(co_genes, all_expressed, gene2go_prop,
                               terms, ns, min_term_size=5, max_term_size=500)
        if df.empty:
            print(f'  No terms tested for {ns_short}')
            continue

        sig = df[df['fdr'] < 0.05]
        print(f'  {len(sig)}/{len(df)} terms significant (FDR<0.05)')
        if len(sig):
            print(df.head(10)[['go_id', 'go_name', 'k', 'n_bg', 'fold_enrichment',
                                'pvalue', 'fdr']].to_string(index=False))

        out_tsv = os.path.join(outdir, f'go_enrichment_{ns_short}_co_retained.tsv')
        df.to_csv(out_tsv, sep='\t', index=False)

        plot_go_dotplot(
            df,
            title=f'GO {ns_short} — co-retained genes (top {min(20, len(sig))} FDR<0.05)',
            outpath=os.path.join(fig_dir, f'go_{ns_short}_dotplot.png'),
        )
        all_rows.append(df.head(20).assign(query='co_retained'))

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(os.path.join(outdir, 'go_enrichment_top20_combined.tsv'),
                        sep='\t', index=False)
        print(f'  Combined top-20 → {outdir}/go_enrichment_top20_combined.tsv')


# ── Analysis 2: gnomAD dosage sensitivity ─────────────────────────────────────

def analysis_gnomad(
    co_genes: set,
    ir_only: set,
    non_ir: set,
    data_dir: str,
    outdir: str,
) -> None:
    candidates = [
        os.path.join(data_dir, 'gnomad', 'gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz'),
        os.path.join(data_dir, 'gnomad', 'gnomad.v2.1.1.lof_metrics.by_gene.txt.gz'),
        os.path.join(data_dir, 'gnomad', 'gnomad.v2.1.1.lof_metrics.by_gene.txt'),
        os.path.join(data_dir, 'gnomad', 'gnomad_gene_constraint.tsv'),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        print(f'  [SKIP] gnomAD constraint file not found in {data_dir}/gnomad/')
        return

    print(f'  Loading gnomAD: {path}')
    open_fn = gzip.open if path.endswith(('.gz', '.bgz')) else open
    with open_fn(path, 'rt') as fh:
        header = fh.readline()
    sep = '\t' if '\t' in header else ','

    with open_fn(path, 'rt') as fh:
        df = pd.read_csv(fh, sep=sep, low_memory=False)

    # Normalise column names (different gnomAD versions vary)
    df.columns = df.columns.str.strip()
    gene_col = next((c for c in df.columns if c in ('gene', 'gene_symbol', 'gene_id')), None)
    if gene_col is None:
        print(f'  [SKIP] cannot find gene column in {path}. Columns: {list(df.columns[:10])}')
        return

    pli_col   = next((c for c in df.columns if c == 'pLI'), None)
    loeuf_col = next((c for c in df.columns
                      if c in ('oe_lof_upper', 'loeuf', 'LOEUF')), None)

    if pli_col is None and loeuf_col is None:
        print(f'  [SKIP] no pLI or LOEUF column found. Columns: {list(df.columns[:20])}')
        return

    # One row per gene: take most constrained transcript (highest pLI)
    sort_col = pli_col if pli_col else loeuf_col
    asc = (sort_col == loeuf_col)  # LOEUF: lower = more constrained
    df_gene = (df.dropna(subset=[gene_col])
                 .sort_values(sort_col, ascending=asc, na_position='last')
                 .groupby(gene_col, as_index=False)
                 .first())
    df_gene = df_gene.rename(columns={gene_col: 'gene'})

    # Assign gene class
    def _cls(g):
        if g in co_genes:   return 'co_retained'
        if g in ir_only:    return 'ir_only'
        return 'non_ir'

    df_gene['gene_class'] = df_gene['gene'].map(_cls)
    out_tsv = os.path.join(outdir, 'gnomad_constraint_by_class.tsv')
    df_gene[['gene', 'gene_class'] +
            [c for c in [pli_col, loeuf_col] if c]].to_csv(
        out_tsv, sep='\t', index=False)

    # Stats + plots
    stat_rows = []
    classes = ['co_retained', 'ir_only', 'non_ir']
    for score_col, score_label, higher_is_bad in [
        (pli_col,   'pLI (LoF intolerance)', True),
        (loeuf_col, 'LOEUF (lower = more constrained)', False),
    ]:
        if score_col is None:
            continue
        groups = {c: df_gene.loc[df_gene['gene_class'] == c, score_col].dropna().values
                  for c in classes}
        print(f'\n  gnomAD {score_col}:')
        for c in classes:
            g = groups[c]
            if len(g):
                print(f'    {c:<14}: n={len(g):>4}  median={np.median(g):.3f}'
                      f'  mean={np.mean(g):.3f}')

        combos = [('co_retained', 'ir_only'),
                  ('co_retained', 'non_ir'),
                  ('ir_only',     'non_ir')]
        for c1, c2 in combos:
            d1, d2 = groups[c1], groups[c2]
            if len(d1) >= 3 and len(d2) >= 3:
                stat, pval = mannwhitneyu(d1, d2, alternative='greater' if higher_is_bad
                                          else 'less')
                stat_rows.append({'score': score_col, 'group1': c1, 'group2': c2,
                                  'n1': len(d1), 'n2': len(d2),
                                  'median1': float(np.median(d1)),
                                  'median2': float(np.median(d2)),
                                  'mw_stat': stat, 'pvalue': pval})
                print(f'    {c1} vs {c2}: MWU p={pval:.3e}')

        # Box plot
        fig, ax = plt.subplots(figsize=(6, 5))
        data = [groups[c] for c in classes]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', lw=1.5))
        for patch, c in zip(bp['boxes'], classes):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.8)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([CLASS_LABELS[c] for c in classes], fontsize=9)
        ax.set_ylabel(score_label, fontsize=9)
        ax.set_title(f'gnomAD {score_col} by IR class', fontsize=10)
        for xi, (c, d) in enumerate(zip(classes, data), 1):
            ax.text(xi, ax.get_ylim()[0], f'n={len(d)}',
                    ha='center', va='top', fontsize=7, color='grey')
        plt.tight_layout()
        fig_dir = os.path.join(outdir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, f'gnomad_{score_col}_boxplot.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        _, fdr, _, _ = multipletests(stat_df['pvalue'].fillna(1.0).values, method='fdr_bh')
        stat_df['fdr'] = fdr
        stat_df.to_csv(os.path.join(outdir, 'gnomad_mwu_tests.tsv'), sep='\t', index=False)
        print(f'  Stats → {outdir}/gnomad_mwu_tests.tsv')


# ── Analysis 3: DepMap essentiality ───────────────────────────────────────────

def analysis_depmap(
    co_genes: set,
    ir_only: set,
    non_ir: set,
    data_dir: str,
    outdir: str,
) -> None:
    candidates = [
        os.path.join(data_dir, 'depmap', 'CRISPRGeneEffect.csv'),
        os.path.join(data_dir, 'depmap', 'Chronos_Combined.csv'),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        print(f'  [SKIP] DepMap gene effect file not found in {data_dir}/depmap/')
        return

    print(f'  Loading DepMap: {path}')
    df = pd.read_csv(path, index_col=0)

    # Columns are "SYMBOL (EntrezID)"; strip to symbol
    col_map = {}
    for col in df.columns:
        sym = col.split(' (')[0].strip() if ' (' in col else col.strip()
        col_map[col] = sym
    df.rename(columns=col_map, inplace=True)

    # Median Chronos score per gene across all cell lines
    medians = df.median(axis=0)
    med_df = pd.DataFrame({'gene': medians.index, 'median_chronos': medians.values})

    def _cls(g):
        if g in co_genes:  return 'co_retained'
        if g in ir_only:   return 'ir_only'
        return 'non_ir'

    med_df['gene_class'] = med_df['gene'].map(_cls)
    med_df.to_csv(os.path.join(outdir, 'depmap_essentiality_by_class.tsv'),
                  sep='\t', index=False)

    classes = ['co_retained', 'ir_only', 'non_ir']
    groups = {c: med_df.loc[med_df['gene_class'] == c, 'median_chronos'].dropna().values
              for c in classes}

    print('\n  DepMap median Chronos score (more negative = more essential):')
    for c in classes:
        g = groups[c]
        if len(g):
            print(f'    {c:<14}: n={len(g):>4}  median={np.median(g):.3f}'
                  f'  mean={np.mean(g):.3f}')

    stat_rows = []
    combos = [('co_retained', 'ir_only'), ('co_retained', 'non_ir'), ('ir_only', 'non_ir')]
    for c1, c2 in combos:
        d1, d2 = groups[c1], groups[c2]
        if len(d1) >= 3 and len(d2) >= 3:
            # co_retained expected to have lower (more negative) scores → alternative='less'
            stat, pval = mannwhitneyu(d1, d2, alternative='less')
            stat_rows.append({'group1': c1, 'group2': c2,
                               'n1': len(d1), 'n2': len(d2),
                               'median1': float(np.median(d1)),
                               'median2': float(np.median(d2)),
                               'mw_stat': stat, 'pvalue': pval})
            print(f'    {c1} vs {c2}: MWU p={pval:.3e}')

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        _, fdr, _, _ = multipletests(stat_df['pvalue'].fillna(1.0).values, method='fdr_bh')
        stat_df['fdr'] = fdr
        stat_df.to_csv(os.path.join(outdir, 'depmap_mwu_tests.tsv'), sep='\t', index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    data = [groups[c] for c in classes]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=1.5))
    for patch, c in zip(bp['boxes'], classes):
        patch.set_facecolor(CLASS_COLORS[c])
        patch.set_alpha(0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], fontsize=9)
    ax.set_ylabel('Median Chronos gene effect score\n(more negative = more essential)',
                  fontsize=9)
    ax.set_title('DepMap essentiality by IR class', fontsize=10)
    ax.axhline(0, lw=0.5, color='grey', linestyle='--')
    for xi, (c, d) in enumerate(zip(classes, data), 1):
        ax.text(xi, ax.get_ylim()[0], f'n={len(d)}',
                ha='center', va='top', fontsize=7, color='grey')
    plt.tight_layout()
    fig_dir = os.path.join(outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, 'depmap_essentiality_boxplot.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure → {outdir}/figures/depmap_essentiality_boxplot.png')


# ── Analysis 4: Constitutive vs cell-type-specific ───────────────────────────

def _go_enrichment_for_subset(
    query: set,
    background: set,
    gene2go_prop: dict | None,
    terms: dict | None,
    outdir: str,
    label: str,
) -> None:
    """Run GO enrichment for a gene subset and save results. Skips if GO not loaded."""
    if gene2go_prop is None or terms is None:
        return
    fig_dir = os.path.join(outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    rows = []
    for ns in ('biological_process', 'molecular_function', 'cellular_component'):
        df = run_go_enrichment(query, background, gene2go_prop, terms, ns,
                               min_term_size=5, max_term_size=500)
        if not df.empty:
            rows.append(df.head(20))
            ns_short = NS_SHORT[ns]
            plot_go_dotplot(
                df,
                title=f'GO {ns_short} — {label}',
                outpath=os.path.join(fig_dir, f'go_{ns_short}_{label.replace(" ", "_")}.png'),
            )
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(os.path.join(outdir, f'go_enrichment_{label.replace(" ", "_")}.tsv'),
                        sep='\t', index=False)


def analysis_constitutive(
    gene_to_cl: dict,
    all_expressed: set,
    data_dir: str,
    outdir: str,
    gene2go_prop: dict | None = None,
    terms: dict | None = None,
    constitutive_min: int = 4,
) -> None:
    n_total_cls = 7  # A549, HEYA8, Hct116, HepG2, K562, MCF7, MCF7-EV

    cat_rows = []
    constitutive, semi, specific = set(), set(), set()

    for gene, cls in gene_to_cl.items():
        n_cls = len(cls)
        if n_cls >= constitutive_min:
            category = 'constitutive'
            constitutive.add(gene)
        elif n_cls >= 2:
            category = 'semi_constitutive'
            semi.add(gene)
        else:
            category = 'cell_type_specific'
            specific.add(gene)
        cat_rows.append({'gene': gene, 'n_cell_lines': n_cls,
                         'cell_lines': ';'.join(sorted(cls)), 'category': category})

    cat_df = pd.DataFrame(cat_rows).sort_values('n_cell_lines', ascending=False)
    cat_df.to_csv(os.path.join(outdir, 'constitutive_categories.tsv'), sep='\t', index=False)

    print(f'\n  Co-retention category counts (out of {n_total_cls} cell lines):')
    for cat, genes in [('constitutive (≥4)',  constitutive),
                       ('semi (2–3)',          semi),
                       ('cell-type-specific (1)', specific)]:
        print(f'    {cat:<26}: {len(genes)} genes')

    # Top constitutive genes by breadth
    top_const = cat_df[cat_df['category'] == 'constitutive'].head(20)
    print(f'\n  Top constitutive co-retention genes (present in ≥{constitutive_min} cell lines):')
    print(top_const[['gene', 'n_cell_lines', 'cell_lines']].to_string(index=False))

    # GO enrichment for constitutive vs all co-retained (background)
    all_co = set(gene_to_cl.keys())
    print('\n  GO enrichment: constitutive vs all co-retained genes...')
    _go_enrichment_for_subset(constitutive, all_co, gene2go_prop, terms,
                               outdir, 'constitutive')
    print('  GO enrichment: cell-type-specific vs all co-retained genes...')
    _go_enrichment_for_subset(specific, all_co, gene2go_prop, terms,
                               outdir, 'cell_type_specific')

    # Stacked bar of category distribution per cell line
    cl_cat: dict = defaultdict(lambda: {'constitutive': 0, 'semi': 0, 'specific': 0})
    for gene, cls in gene_to_cl.items():
        cat = ('constitutive' if gene in constitutive
               else 'semi' if gene in semi else 'specific')
        for cl in cls:
            cl_cat[cl][cat] += 1

    fig, ax = plt.subplots(figsize=(9, 5))
    cell_lines_sorted = sorted(cl_cat.keys())
    x = np.arange(len(cell_lines_sorted))
    w = 0.5
    bottoms = np.zeros(len(cell_lines_sorted))
    cat_colors = {'constitutive': 'darkred', 'semi': 'coral', 'specific': 'lightgrey'}
    for cat, color in cat_colors.items():
        vals = [cl_cat[cl][cat] for cl in cell_lines_sorted]
        ax.bar(x, vals, w, bottom=bottoms, label=cat.replace('_', ' '), color=color,
               alpha=0.85)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(cell_lines_sorted, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Number of co-retained genes', fontsize=9)
    ax.set_title(f'Co-retention category per cell line (constitutive ≥{constitutive_min} cls)',
                 fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig_dir = os.path.join(outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, 'constitutive_category_per_cellline.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure → {outdir}/figures/constitutive_category_per_cellline.png')


# ── Analysis 5: CORUM protein complex membership ──────────────────────────────

def analysis_corum(
    co_genes: set,
    ir_only: set,
    non_ir: set,
    data_dir: str,
    outdir: str,
) -> None:
    candidates = [
        os.path.join(data_dir, 'corum', 'allComplexes.txt'),
        os.path.join(data_dir, 'corum', 'humanComplexes.txt'),
        os.path.join(data_dir, 'corum', 'allComplexes.txt.zip'),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        print(f'  [SKIP] CORUM file not found in {data_dir}/corum/')
        return

    print(f'  Loading CORUM: {path}')
    if path.endswith('.zip'):
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                inner = [n for n in zf.namelist() if n.endswith('.txt')]
                if not inner:
                    print('  [SKIP] No .txt inside CORUM zip')
                    return
                with zf.open(inner[0]) as fh:
                    df = pd.read_csv(fh, sep='\t')
        except zipfile.BadZipFile:
            print(f'  [SKIP] {path} is not a valid zip (possibly corrupt or HTML). '
                  'Re-download the source data file.')
            return
    else:
        df = pd.read_csv(path, sep='\t')

    # Filter for human entries
    org_col = next((c for c in df.columns if c.lower() in ('organism', 'organism_ncbi_id')),
                   None)
    if org_col:
        df = df[df[org_col].astype(str).str.contains('Human|9606', case=False, na=False)]
    print(f'  {len(df)} human complexes')

    # Find the gene list column (CORUM: "subunits(Gene name)"; EBI-derived: same)
    gene_col = next((c for c in df.columns
                     if 'gene name' in c.lower() or 'subunit' in c.lower()), None)
    if gene_col is None:
        print(f'  [SKIP] Cannot find gene list column. Columns: {list(df.columns[:10])}')
        return

    complex_genes: set = set()
    complex_to_genes: dict = {}
    name_col = next((c for c in df.columns
                     if 'complexname' in c.lower() or 'complex name' in c.lower()
                     or c.lower() == 'complexname'), None)

    for _, row in df.iterrows():
        raw = str(row[gene_col])
        if raw in ('nan', ''):
            continue
        genes_in_complex = {g.strip() for g in raw.split(';') if g.strip()}
        complex_genes.update(genes_in_complex)
        name = str(row[name_col]) if name_col else f'complex_{_}'
        complex_to_genes[name] = genes_in_complex

    print(f'  {len(complex_to_genes)} complexes, {len(complex_genes)} unique genes')

    # Background = union of all three classes
    background = co_genes | ir_only | non_ir
    classes = {
        'co_retained': co_genes,
        'ir_only':     ir_only,
        'non_ir':      non_ir,
    }

    stat_rows = []
    for cls_name, cls_genes in classes.items():
        in_complex  = len(cls_genes & complex_genes)
        not_complex = len(cls_genes) - in_complex
        frac = in_complex / len(cls_genes) * 100 if cls_genes else 0
        stat_rows.append({
            'gene_class':   cls_name,
            'n_genes':      len(cls_genes),
            'in_complex':   in_complex,
            'not_complex':  not_complex,
            'pct_in_complex': frac,
        })
        print(f'  {cls_name:<14}: {in_complex}/{len(cls_genes)} ({frac:.1f}%) in CORUM complex')

    # Fisher: co_retained vs ir_only
    co_row = next(r for r in stat_rows if r['gene_class'] == 'co_retained')
    ir_row = next(r for r in stat_rows if r['gene_class'] == 'ir_only')
    ni_row = next(r for r in stat_rows if r['gene_class'] == 'non_ir')

    for ref_name, ref_row in [('ir_only', ir_row), ('non_ir', ni_row)]:
        table = [
            [co_row['in_complex'], co_row['not_complex']],
            [ref_row['in_complex'], ref_row['not_complex']],
        ]
        or_, pval = fisher_exact(table, alternative='greater')
        print(f'  co_retained vs {ref_name}: OR={or_:.3f}  p={pval:.3e}')
        stat_rows.append({
            'gene_class': f'fisher_co_vs_{ref_name}',
            'n_genes': None, 'in_complex': None, 'not_complex': None,
            'pct_in_complex': None,
            'fisher_or': or_, 'fisher_pvalue': pval,
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(os.path.join(outdir, 'corum_complex_membership.tsv'),
                   sep='\t', index=False)

    # Test enrichment in specific well-known complexes
    complex_categories = {
        'Ribosome':    ['ribosom', 'rpl', 'rps'],
        'Spliceosome': ['spliceosom', 'snrnp', 'prp'],
        'Proteasome':  ['proteasom', 'psm'],
        'Exosome':     ['exosom'],
        'Mediator':    ['mediator'],
    }
    cat_rows = []
    for cat_label, keywords in complex_categories.items():
        cat_genes: set = set()
        for cname, cgenes in complex_to_genes.items():
            if any(kw in cname.lower() for kw in keywords):
                cat_genes.update(cgenes)
        if not cat_genes:
            continue
        for cls_name, cls_genes in classes.items():
            n_in  = len(cls_genes & cat_genes)
            n_tot = len(cls_genes)
            cat_rows.append({'complex_category': cat_label, 'gene_class': cls_name,
                              'n_in': n_in, 'n_total': n_tot,
                              'pct': n_in / n_tot * 100 if n_tot else 0})

    if cat_rows:
        cat_df = pd.DataFrame(cat_rows)
        cat_df.to_csv(os.path.join(outdir, 'corum_by_complex_category.tsv'),
                      sep='\t', index=False)
        print(f'  Per-category → {outdir}/corum_by_complex_category.tsv')

        # Grouped bar chart: complex categories × gene class
        cats = cat_df['complex_category'].unique()
        clss = ['co_retained', 'ir_only', 'non_ir']
        x = np.arange(len(cats))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.2), 5))
        for i, cls_name in enumerate(clss):
            sub = cat_df[cat_df['gene_class'] == cls_name].set_index('complex_category')
            vals = [sub.loc[c, 'pct'] if c in sub.index else 0 for c in cats]
            ax.bar(x + (i - 1) * w, vals, w, label=CLASS_LABELS[cls_name],
                   color=CLASS_COLORS[cls_name], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('% genes in complex category', fontsize=9)
        ax.set_title('CORUM protein complex membership by gene class', fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig_dir = os.path.join(outdir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, 'corum_complex_categories.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Figure → {outdir}/figures/corum_complex_categories.png')

    # Overall membership bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    cls_order = ['co_retained', 'ir_only', 'non_ir']
    pcts = [next(r['pct_in_complex'] for r in stat_rows
                 if r['gene_class'] == c) for c in cls_order]
    bars = ax.bar([CLASS_LABELS[c] for c in cls_order], pcts,
                  color=[CLASS_COLORS[c] for c in cls_order], alpha=0.85)
    for bar, pct, cls in zip(bars, pcts, cls_order):
        n = len(classes[cls])
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{pct:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('% genes in any CORUM complex', fontsize=9)
    ax.set_title('CORUM protein complex membership', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'figures', 'corum_membership_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure → {outdir}/figures/corum_membership_bar.png')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', default='results',
                        help='Root results directory (default: results)')
    parser.add_argument('--data-dir', default='data',
                        help='Root data directory (default: data)')
    parser.add_argument('--outdir', default='results/mechanistic/gene_function',
                        help='Output directory')
    parser.add_argument('--fdr-threshold', type=float, default=0.05)
    parser.add_argument('--ir-threshold',  type=float, default=0.05)
    parser.add_argument('--constitutive-min', type=int, default=4,
                        help='Min cell lines for "constitutive" label (default: 4)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'figures'), exist_ok=True)

    print('=' * 60)
    print('Gene function analysis')
    print(f'  results_dir       : {args.results_dir}')
    print(f'  data_dir          : {args.data_dir}')
    print(f'  outdir            : {args.outdir}')
    print(f'  fdr_threshold     : {args.fdr_threshold}')
    print(f'  ir_threshold      : {args.ir_threshold}')
    print(f'  constitutive_min  : {args.constitutive_min}')
    print('=' * 60)

    # Build gene classes (required by all analyses)
    print('\n[0] Building gene classes...')
    co_genes, ir_only, non_ir, all_expressed, gene_to_cl = build_gene_classes(
        args.results_dir, args.fdr_threshold, args.ir_threshold)

    # Shared GO data (loaded once, reused by analyses 1 and 4)
    terms_shared: dict | None = None
    gene2go_prop_shared: dict | None = None

    obo_path  = os.path.join(args.data_dir, 'go', 'go-basic.obo')
    gaf_paths = [os.path.join(args.data_dir, 'go', 'goa_human.gaf.gz'),
                 os.path.join(args.data_dir, 'go', 'goa_human.gaf')]
    gaf_path  = next((p for p in gaf_paths if os.path.isfile(p)), None)

    if os.path.isfile(obo_path) and gaf_path:
        print('\n  Pre-loading GO data (shared by analyses 1 and 4)...')
        terms_shared, ancestors = parse_obo(obo_path)
        gene2go_raw = load_gaf(gaf_path, gene_universe=all_expressed)
        gene2go_prop_shared = propagate_annotations(gene2go_raw, ancestors)

    # ── Analysis 1: GO enrichment ────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('[1] GO enrichment — co-retained vs all expressed')
    print('=' * 60)
    if terms_shared is not None:
        all_rows = []
        fig_dir  = os.path.join(args.outdir, 'figures')
        for ns in ('biological_process', 'molecular_function', 'cellular_component'):
            ns_short = NS_SHORT[ns]
            print(f'  {ns_short}...')
            df = run_go_enrichment(co_genes, all_expressed, gene2go_prop_shared,
                                   terms_shared, ns, min_term_size=5, max_term_size=500)
            if df.empty:
                continue
            sig = df[df['fdr'] < 0.05]
            print(f'  {len(sig)}/{len(df)} significant (FDR<0.05)')
            if len(sig):
                print(df.head(10)[['go_id', 'go_name', 'k', 'n_bg',
                                   'fold_enrichment', 'fdr']].to_string(index=False))
            df.to_csv(os.path.join(args.outdir,
                                   f'go_enrichment_{ns_short}_co_retained.tsv'),
                      sep='\t', index=False)
            plot_go_dotplot(
                df,
                title=f'GO {ns_short} — co-retained genes',
                outpath=os.path.join(fig_dir, f'go_{ns_short}_co_retained_dotplot.png'),
            )
            all_rows.append(df.head(20).assign(namespace_short=ns_short))
        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            combined.to_csv(os.path.join(args.outdir, 'go_enrichment_top20_combined.tsv'),
                            sep='\t', index=False)
    else:
        print('  [SKIP] GO data not available')

    # ── Analysis 2: gnomAD ───────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('[2] gnomAD dosage sensitivity')
    print('=' * 60)
    analysis_gnomad(co_genes, ir_only, non_ir, args.data_dir, args.outdir)

    # ── Analysis 3: DepMap ───────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('[3] DepMap essentiality')
    print('=' * 60)
    analysis_depmap(co_genes, ir_only, non_ir, args.data_dir, args.outdir)

    # ── Analysis 4: Constitutive vs cell-type-specific ───────────────────────
    print('\n' + '=' * 60)
    print('[4] Constitutive vs cell-type-specific co-retention')
    print('=' * 60)
    analysis_constitutive(
        gene_to_cl, all_expressed, args.data_dir, args.outdir,
        gene2go_prop=gene2go_prop_shared,
        terms=terms_shared,
        constitutive_min=args.constitutive_min,
    )

    # ── Analysis 5: CORUM ────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('[5] CORUM protein complex membership')
    print('=' * 60)
    analysis_corum(co_genes, ir_only, non_ir, args.data_dir, args.outdir)

    print('\n' + '=' * 60)
    print('All analyses complete.')
    print(f'Outputs in {args.outdir}')
    print('=' * 60)
    import subprocess
    subprocess.run(['ls', '-lh', args.outdir], check=False)


if __name__ == '__main__':
    main()

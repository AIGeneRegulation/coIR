"""
detained_intron_overlap.py — Overlap co-retained introns against the Boutz 2015 detained intron catalog.

Tests whether co-retained introns are enriched for detained introns relative to all IR introns,
using Fisher's exact test. Reports overlap statistics per cell type (HeLa, HepG2, HUVEC).

Usage:
    python src/detained_intron_overlap.py \
        --coretention results/pilot_v2/<sample>_coretention.tsv \
        --ir-events results/pilot_v2/<sample>_ir_events.tsv \
        --di-catalog data/detained_introns/ \
        --outdir results/pilot_v2/
"""

import argparse
import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


_HUMAN_DI_DIRS = [
    "data/detained_introns_human",
    "data/detained_introns",
]


def load_di_catalog(di_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load detained intron files from the Boutz 2015 supplementary data (GSE57231).

    Handles two formats:
      1. Boutz IntronQuantification TSV: columns intron_id, chr, start, end, ..., DI
         Filters rows where DI == "DI".
      2. Generic BED/TSV with chrom/start/end columns.

    NOTE: GSE57231 data is MOUSE (gene names like 'Arhgef1'). Coordinates are mm10,
    not hg38. Overlaps with human introns will be ~0 by design.
    Returns dict of {cell_type: DataFrame with chrom/start/end columns}.
    """
    catalogs = {}
    candidates = (
        glob.glob(os.path.join(di_dir, "*.bed")) +
        glob.glob(os.path.join(di_dir, "*.txt")) +
        glob.glob(os.path.join(di_dir, "*.tsv")) +
        glob.glob(os.path.join(di_dir, "*.csv"))
    )

    for fpath in sorted(candidates):
        fname = os.path.basename(fpath).lower()
        if any(skip in fname for skip in ('readme', 'index', 'html', 'raw', 'accession', 'needed',
                                           'bedgraph', 'miso', 'robots', 'filelist')):
            continue

        try:
            for sep in ('\t', ','):
                try:
                    df = pd.read_csv(fpath, sep=sep, comment='#', low_memory=False)
                    if len(df.columns) >= 3:
                        break
                except Exception:
                    continue

            df.columns = [c.strip().lower() for c in df.columns]

            # Boutz human BED (from 14_download_boutz_human.sh):
            # chrom, start, end, name, score/n_comparisons, [strand, significant, ...]
            if 'chrom' in df.columns and 'start' in df.columns and 'end' in df.columns:
                df['start'] = pd.to_numeric(df['start'], errors='coerce')
                df['end']   = pd.to_numeric(df['end'],   errors='coerce')
                df = df.dropna(subset=['chrom', 'start', 'end'])
                df['start'] = df['start'].astype(int)
                df['end']   = df['end'].astype(int)
                # Label by file name: union, highconf, or per-comparison
                cell_type = 'HeLa_union'
                if 'highconf' in fname:
                    cell_type = 'HeLa_highconf'
                elif 'hepg2' in fname:
                    cell_type = 'HepG2'
                elif 'huvec' in fname:
                    cell_type = 'HUVEC'
                elif any(kd in fname for kd in ('4a3', 'mln51', 'upf1', 'y14')):
                    kd = next(k for k in ('4a3', 'mln51', 'upf1', 'y14') if k in fname)
                    cell_type = f'HeLa_{kd.upper()}'
                catalogs[cell_type] = df
                print(f"  Loaded {len(df)} DIs from {os.path.basename(fpath)} [{cell_type}] (human)")
                continue

            # Boutz IntronQuantification format: has 'di' column and 'chr'/'start'/'end'
            if 'di' in df.columns and 'chr' in df.columns:
                di_df = df[df['di'] == 'DI'].copy()
                di_df = di_df.rename(columns={'chr': 'chrom'})
                di_df['start'] = pd.to_numeric(di_df['start'], errors='coerce')
                di_df['end'] = pd.to_numeric(di_df['end'], errors='coerce')
                di_df = di_df.dropna(subset=['chrom', 'start', 'end'])
                di_df['start'] = di_df['start'].astype(int)
                di_df['end'] = di_df['end'].astype(int)

                # Detect mouse data from gene names (mouse: lowercase e.g. 'Arhgef1_intron_1')
                if 'intron_id' in di_df.columns:
                    sample_id = di_df['intron_id'].iloc[0] if len(di_df) > 0 else ""
                    gene_part = sample_id.split('_')[0]
                    is_mouse = (len(gene_part) > 1 and gene_part[0].isupper()
                                and gene_part[1:].islower())
                    if is_mouse:
                        print(f"  WARNING: {os.path.basename(fpath)} appears to be MOUSE data "
                              f"(e.g. '{sample_id}'). Coordinates are mm10, not hg38.")
                        print(f"  Overlap with human introns will be ~0.")

                cell_type = "Boutz2015_mouse"
                catalogs[cell_type] = di_df
                print(f"  Loaded {len(di_df)} detained introns (DI==DI) from "
                      f"{os.path.basename(fpath)} [{cell_type}]")
                continue

            # Generic BED/TSV format
            col_map = {}
            for c in df.columns:
                if c in ('chr', 'chrom', 'chromosome', 'seqname', 'seqnames'):
                    col_map[c] = 'chrom'
                elif c in ('start', 'chromstart', 'txstart', 'intron_start', 'begin'):
                    col_map[c] = 'start'
                elif c in ('end', 'chromend', 'txend', 'intron_end', 'stop'):
                    col_map[c] = 'end'
            df = df.rename(columns=col_map)

            if 'chrom' not in df.columns and len(df.columns) >= 3:
                df = df.rename(columns={df.columns[0]: 'chrom',
                                        df.columns[1]: 'start', df.columns[2]: 'end'})

            if not {'chrom', 'start', 'end'}.issubset(df.columns):
                continue

            df['start'] = pd.to_numeric(df['start'], errors='coerce')
            df['end'] = pd.to_numeric(df['end'], errors='coerce')
            df = df.dropna(subset=['chrom', 'start', 'end'])
            df['start'] = df['start'].astype(int)
            df['end'] = df['end'].astype(int)

            cell_type = os.path.splitext(os.path.basename(fpath))[0]
            for ct in ('hela', 'hepg2', 'huvec'):
                if ct in fname:
                    cell_type = ct.upper()
                    break
            catalogs[cell_type] = df
            print(f"  Loaded {len(df)} DIs from {os.path.basename(fpath)} [{cell_type}]")

        except Exception as e:
            print(f"  Warning: could not load {fpath}: {e}")

    return catalogs


def intron_key(chrom: str, start: int, end: int) -> str:
    # Normalise chr prefix
    chrom = chrom if chrom.startswith('chr') else 'chr' + chrom
    return f"{chrom}:{start}-{end}"


def build_interval_set(df: pd.DataFrame) -> set:
    """Build set of intron keys, normalising chr prefix."""
    keys = set()
    for _, row in df.iterrows():
        chrom = str(row['chrom'])
        chrom = chrom if chrom.startswith('chr') else 'chr' + chrom
        keys.add(f"{chrom}:{int(row['start'])}-{int(row['end'])}")
    return keys


def overlaps_any(chrom: str, start: int, end: int, di_df: pd.DataFrame,
                 di_chroms: dict) -> bool:
    """Check if intron [start, end) overlaps any DI on the same chromosome."""
    chrom_norm = chrom if chrom.startswith('chr') else 'chr' + chrom
    chrom_di = di_chroms.get(chrom_norm)
    if chrom_di is None:
        return False
    # Any DI that overlaps [start, end)
    mask = (chrom_di['end'] > start) & (chrom_di['start'] < end)
    return mask.any()


def compute_overlap(coret_df: pd.DataFrame, ir_df: pd.DataFrame,
                    di_df: pd.DataFrame, fdr_threshold: float = 0.05) -> dict:
    """
    Fisher's exact test: are co-retained introns enriched for detained introns?

    2x2 table:
                    DI      not-DI
    co-retained    |  a  |   b  |
    IR-only        |  c  |   d  |
    """
    # Pre-index DI catalog by chromosome
    di_chroms = {
        chrom: grp for chrom, grp in di_df.groupby('chrom')
    }

    # Significant co-retention pairs → set of intron coordinate strings
    sig = coret_df[coret_df['fdr'] < fdr_threshold]
    coret_introns = set()
    for _, row in sig.iterrows():
        for col in ('intron_a', 'intron_b'):
            coret_introns.add(row[col])  # format: "chrom:start-end"

    # All IR introns → coordinate strings
    all_ir_introns = set()
    for _, row in ir_df.iterrows():
        key = intron_key(str(row['chrom']), int(row['intron_start']), int(row['intron_end']))
        all_ir_introns.add(key)

    ir_only_introns = all_ir_introns - coret_introns

    def count_di(intron_set):
        n_di = 0
        for key in intron_set:
            parts = key.split(':')
            chrom = parts[0]
            start, end = map(int, parts[1].split('-'))
            if overlaps_any(chrom, start, end, di_df, di_chroms):
                n_di += 1
        return n_di

    n_coret = len(coret_introns)
    n_ir_only = len(ir_only_introns)
    coret_di = count_di(coret_introns)
    ir_only_di = count_di(ir_only_introns)

    table = np.array([
        [coret_di, n_coret - coret_di],
        [ir_only_di, n_ir_only - ir_only_di],
    ])
    odds_ratio, pvalue = stats.fisher_exact(table, alternative='greater')

    return {
        'n_coretained_introns': n_coret,
        'n_ir_only_introns': n_ir_only,
        'n_detained_in_coretained': coret_di,
        'n_detained_in_ir_only': ir_only_di,
        'pct_detained_coretained': coret_di / n_coret * 100 if n_coret else 0,
        'pct_detained_ir_only': ir_only_di / n_ir_only * 100 if n_ir_only else 0,
        'odds_ratio': odds_ratio,
        'pvalue': pvalue,
        'table': table.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Detained intron enrichment in co-retention")
    parser.add_argument("--coretention", required=True)
    parser.add_argument("--ir-events", required=True)
    parser.add_argument("--di-catalog", default=None,
                        help="Directory with Boutz DI files (default: auto-detects "
                             "data/detained_introns_human/ then data/detained_introns/)")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sample_name = os.path.basename(args.coretention).replace('_coretention.tsv', '')

    print("Loading co-retention results...")
    coret_df = pd.read_csv(args.coretention, sep='\t')
    print(f"  {len(coret_df)} pairs, {(coret_df['fdr'] < args.fdr_threshold).sum()} significant")

    print("Loading IR events...")
    ir_df = pd.read_csv(args.ir_events, sep='\t')
    print(f"  {len(ir_df)} IR events")

    # Auto-detect DI catalog directory
    di_dir = args.di_catalog
    if di_dir is None:
        for candidate in _HUMAN_DI_DIRS:
            if os.path.isdir(candidate) and any(
                    f.endswith('.bed') or f.endswith('.tsv')
                    for f in os.listdir(candidate)):
                di_dir = candidate
                print(f"  Auto-detected DI catalog: {di_dir}")
                break
    if di_dir is None:
        print("  ERROR: no DI catalog directory found.")
        print("  Provide --di-dir pointing to the Boutz et al. detained intron catalog.")
        return

    print("Loading detained intron catalog...")
    catalogs = load_di_catalog(di_dir)
    if not catalogs:
        print(f"  ERROR: no DI files found in {di_dir}")
        print("  Download the Boutz et al. detained intron catalog and point --di-dir to it.")
        return

    rows = []
    for cell_type, di_df in catalogs.items():
        print(f"\nTesting overlap with {cell_type} detained introns ({len(di_df)} DIs)...")
        result = compute_overlap(coret_df, ir_df, di_df, args.fdr_threshold)
        result['di_cell_type'] = cell_type
        result['sample'] = sample_name
        rows.append(result)

        print(f"  Co-retained introns with DI: {result['n_detained_in_coretained']}"
              f" / {result['n_coretained_introns']} ({result['pct_detained_coretained']:.1f}%)")
        print(f"  IR-only introns with DI:     {result['n_detained_in_ir_only']}"
              f" / {result['n_ir_only_introns']} ({result['pct_detained_ir_only']:.1f}%)")
        print(f"  Odds ratio: {result['odds_ratio']:.2f}  p={result['pvalue']:.3e}")

    result_df = pd.DataFrame(rows)
    out_tsv = os.path.join(args.outdir, f'{sample_name}_di_overlap.tsv')
    result_df.drop(columns=['table']).to_csv(out_tsv, sep='\t', index=False)
    print(f"\nSaved: {out_tsv}")

    # Simple bar chart
    if len(rows) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(rows))
        w = 0.35
        ax.bar(x - w/2, [r['pct_detained_coretained'] for r in rows], w,
               label='Co-retained introns', color='coral')
        ax.bar(x + w/2, [r['pct_detained_ir_only'] for r in rows], w,
               label='IR-only introns', color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels([r['di_cell_type'] for r in rows])
        ax.set_ylabel('% overlapping detained introns')
        ax.set_title('Detained intron enrichment in co-retained introns')
        ax.legend()
        for i, r in enumerate(rows):
            sig = '*' if r['pvalue'] < 0.05 else 'ns'
            ax.text(i, max(r['pct_detained_coretained'], r['pct_detained_ir_only']) + 0.5,
                    f"OR={r['odds_ratio']:.1f}\n{sig}", ha='center', fontsize=8)
        plt.tight_layout()
        out_fig = os.path.join(args.outdir, 'figures', f'{sample_name}_di_overlap.png')
        os.makedirs(os.path.dirname(out_fig), exist_ok=True)
        fig.savefig(out_fig, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()

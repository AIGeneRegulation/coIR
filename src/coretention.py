"""
coretention.py — Single-molecule co-retention analysis for long-read RNA-seq.

This is the core novel analysis: for genes with multiple retained introns,
do those introns tend to be co-retained on the SAME molecule?

The original IRFinder paper (Middleton et al. 2017 Genome Biology) showed
that retained introns cluster within genes — often adjacent introns are
co-retained. But with short reads, it was impossible to determine whether
they were retained on the same transcript or in different molecules.

Long reads resolve this directly: a single read spanning two introns either
has both spliced, both retained, or one of each. By tallying these four
states per intron pair, we can test for non-random co-retention.

Statistical approach:
  For each pair of introns (i, j) in the same gene:
    - Count reads spanning both: n_both_spliced, n_i_retained_j_spliced,
      n_i_spliced_j_retained, n_both_retained
    - Test for association with Fisher's exact test
    - Compute phi coefficient (correlation between retention states)
    - FDR-correct across all tested pairs
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pysam
from scipy import stats

from ir_utils import Intron, _classify_read_at_intron, _make_chrom_map


@dataclass
class IntronPairCoretention:
    """Results for a single pair of introns tested for co-retention."""
    gene_id: str
    gene_name: str
    chrom: str
    intron_a_start: int
    intron_a_end: int
    intron_a_index: int
    intron_b_start: int
    intron_b_end: int
    intron_b_index: int
    # 2x2 contingency table
    both_spliced: int
    a_retained_b_spliced: int
    a_spliced_b_retained: int
    both_retained: int
    # Statistics
    total_reads: int
    ir_ratio_a: float
    ir_ratio_b: float
    phi_coefficient: float
    fisher_pvalue: float
    fisher_odds_ratio: float
    adjacent: bool  # True if introns are separated by one exon


def find_coretention_pairs(introns: List[Intron],
                            max_pair_distance: int = 10) -> List[Tuple[Intron, Intron]]:
    """
    Identify intron pairs within the same gene to test for co-retention.
    
    Args:
        introns: List of Intron objects.
        max_pair_distance: Maximum number of introns apart to test.
                          1 = adjacent only, higher = more distant pairs.
    
    Returns:
        List of (intron_a, intron_b) tuples to test.
    """
    # Group introns by gene
    gene_introns: Dict[str, List[Intron]] = defaultdict(list)
    for intron in introns:
        gene_introns[intron.gene_id].append(intron)
    
    pairs = []
    for gene_id, gene_int_list in gene_introns.items():
        if len(gene_int_list) < 2:
            continue
        
        # Sort by genomic position
        sorted_introns = sorted(gene_int_list, key=lambda x: x.start)
        
        for i in range(len(sorted_introns)):
            for j in range(i + 1, min(i + max_pair_distance + 1, len(sorted_introns))):
                a, b = sorted_introns[i], sorted_introns[j]
                overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
                if overlap / min(a.length, b.length) > 0.5:
                    continue
                pairs.append((a, b))
    
    return pairs


def analyze_coretention(bam_path: str,
                         intron_pairs: List[Tuple[Intron, Intron]],
                         min_reads: int = 10,
                         min_overhang: int = 10) -> List[IntronPairCoretention]:
    """
    For each intron pair, classify spanning reads into the 2x2 contingency table.
    
    A read must span BOTH introns entirely to be counted. This means:
    - read.reference_start < intron_a.start - min_overhang
    - read.reference_end > intron_b.end + min_overhang
    
    Args:
        bam_path: Path to sorted, indexed BAM file.
        intron_pairs: List of (intron_a, intron_b) tuples.
        min_reads: Minimum spanning reads to report a pair.
        min_overhang: Minimum overhang past intron boundaries.
    
    Returns:
        List of IntronPairCoretention results.
    """
    bamfile = pysam.AlignmentFile(bam_path, "rb")
    all_introns = [ia for ia, _ in intron_pairs] + [ib for _, ib in intron_pairs]
    chrom_map = _make_chrom_map(bamfile.references, all_introns)
    results = []

    print(f"Analyzing co-retention for {len(intron_pairs)} intron pairs...")

    for idx, (intron_a, intron_b) in enumerate(intron_pairs):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(intron_pairs)} pairs...")

        # intron_a should be upstream of intron_b
        if intron_a.start > intron_b.start:
            intron_a, intron_b = intron_b, intron_a

        bam_chrom = chrom_map.get(intron_a.chrom)
        if bam_chrom is None:
            continue

        # Region that a read must span
        region_start = intron_a.start - min_overhang
        region_end = intron_b.end + min_overhang

        # Skip if the span is too large for realistic read lengths
        span = region_end - region_start
        if span > 50000:  # most long reads are <50kb
            continue

        both_spliced = 0
        a_ret_b_spl = 0
        a_spl_b_ret = 0
        both_retained = 0

        try:
            reads = bamfile.fetch(
                bam_chrom,
                max(0, region_start),
                region_end
            )
        except (ValueError, KeyError):
            continue
        
        for read in reads:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            
            # Read must span both introns entirely
            if read.reference_start > region_start:
                continue
            if read.reference_end is None or read.reference_end < region_end:
                continue
            
            # Classify at each intron
            state_a = _classify_read_at_intron(read, intron_a.start, intron_a.end)
            state_b = _classify_read_at_intron(read, intron_b.start, intron_b.end)
            
            if state_a == 'ambiguous' or state_b == 'ambiguous':
                continue
            
            if state_a == 'spliced' and state_b == 'spliced':
                both_spliced += 1
            elif state_a == 'retained' and state_b == 'spliced':
                a_ret_b_spl += 1
            elif state_a == 'spliced' and state_b == 'retained':
                a_spl_b_ret += 1
            elif state_a == 'retained' and state_b == 'retained':
                both_retained += 1
        
        total = both_spliced + a_ret_b_spl + a_spl_b_ret + both_retained
        if total < min_reads:
            continue
        
        # Compute per-intron IR ratios from the spanning reads
        a_retained_total = a_ret_b_spl + both_retained
        a_spliced_total = both_spliced + a_spl_b_ret
        b_retained_total = a_spl_b_ret + both_retained
        b_spliced_total = both_spliced + a_ret_b_spl
        
        ir_a = a_retained_total / total if total > 0 else 0
        ir_b = b_retained_total / total if total > 0 else 0
        
        # Fisher's exact test on the 2x2 table
        table = np.array([[both_spliced, a_spl_b_ret],
                          [a_ret_b_spl, both_retained]])
        
        try:
            odds_ratio, pvalue = stats.fisher_exact(table, alternative='two-sided')
        except ValueError:
            odds_ratio, pvalue = np.nan, 1.0
        
        # Phi coefficient (correlation for binary variables)
        phi = _phi_coefficient(both_spliced, a_ret_b_spl, a_spl_b_ret, both_retained)
        
        # Are these adjacent introns? (separated by exactly one exon)
        adjacent = (intron_b.intron_index - intron_a.intron_index == 1 and
                    intron_a.gene_id == intron_b.gene_id)
        
        results.append(IntronPairCoretention(
            gene_id=intron_a.gene_id,
            gene_name=intron_a.gene_name,
            chrom=intron_a.chrom,
            intron_a_start=intron_a.start,
            intron_a_end=intron_a.end,
            intron_a_index=intron_a.intron_index,
            intron_b_start=intron_b.start,
            intron_b_end=intron_b.end,
            intron_b_index=intron_b.intron_index,
            both_spliced=both_spliced,
            a_retained_b_spliced=a_ret_b_spl,
            a_spliced_b_retained=a_spl_b_ret,
            both_retained=both_retained,
            total_reads=total,
            ir_ratio_a=ir_a,
            ir_ratio_b=ir_b,
            phi_coefficient=phi,
            fisher_pvalue=pvalue,
            fisher_odds_ratio=odds_ratio if not np.isinf(odds_ratio) else 999.0,
            adjacent=adjacent,
        ))
    
    bamfile.close()
    
    # FDR correction
    if results:
        pvalues = [r.fisher_pvalue for r in results]
        fdr = _benjamini_hochberg(pvalues)
        for r, q in zip(results, fdr):
            r.fdr = q  # type: ignore
    
    print(f"  {len(results)} pairs passed filters (min {min_reads} spanning reads)")
    return results


_CORET_COLUMNS = [
    'gene_id', 'gene_name', 'chrom', 'intron_a', 'intron_b',
    'intron_a_index', 'intron_b_index', 'adjacent',
    'both_spliced', 'a_ret_b_spl', 'a_spl_b_ret', 'both_retained',
    'total_spanning_reads', 'ir_ratio_a', 'ir_ratio_b',
    'phi_coefficient', 'fisher_pvalue', 'fisher_odds_ratio', 'fdr',
]


def coretention_to_dataframe(results: List[IntronPairCoretention]) -> pd.DataFrame:
    """Convert co-retention results to DataFrame."""
    if not results:
        return pd.DataFrame(columns=_CORET_COLUMNS)
    rows = []
    for r in results:
        rows.append({
            'gene_id': r.gene_id,
            'gene_name': r.gene_name,
            'chrom': r.chrom,
            'intron_a': f"{r.chrom}:{r.intron_a_start}-{r.intron_a_end}",
            'intron_b': f"{r.chrom}:{r.intron_b_start}-{r.intron_b_end}",
            'intron_a_index': r.intron_a_index,
            'intron_b_index': r.intron_b_index,
            'adjacent': r.adjacent,
            'both_spliced': r.both_spliced,
            'a_ret_b_spl': r.a_retained_b_spliced,
            'a_spl_b_ret': r.a_spliced_b_retained,
            'both_retained': r.both_retained,
            'total_spanning_reads': r.total_reads,
            'ir_ratio_a': r.ir_ratio_a,
            'ir_ratio_b': r.ir_ratio_b,
            'phi_coefficient': r.phi_coefficient,
            'fisher_pvalue': r.fisher_pvalue,
            'fisher_odds_ratio': r.fisher_odds_ratio,
            'fdr': getattr(r, 'fdr', np.nan),
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('fisher_pvalue')
    return df


def summarize_coretention(df: pd.DataFrame, fdr_threshold: float = 0.05) -> dict:
    """
    Produce summary statistics for the co-retention analysis.
    
    Returns dict with:
    - total_pairs_tested
    - significant_pairs (FDR < threshold)
    - positive_coretention (phi > 0, significant)
    - negative_coretention (phi < 0, significant, i.e. anti-correlated)
    - adjacent_enrichment (are adjacent pairs more likely co-retained?)
    - genes_with_coretention
    """
    sig = df[df['fdr'] < fdr_threshold]
    
    summary = {
        'total_pairs_tested': len(df),
        'significant_pairs': len(sig),
        'positive_coretention': len(sig[sig['phi_coefficient'] > 0]),
        'negative_coretention': len(sig[sig['phi_coefficient'] < 0]),
        'genes_with_coretention': sig['gene_id'].nunique(),
        'median_phi_significant': sig['phi_coefficient'].median() if len(sig) > 0 else np.nan,
    }
    
    # Test whether adjacent pairs are enriched among significant results
    if len(df) > 0 and 'adjacent' in df.columns:
        adj_tested = df['adjacent'].sum()
        nonadj_tested = (~df['adjacent']).sum()
        adj_sig = sig['adjacent'].sum() if len(sig) > 0 else 0
        nonadj_sig = (~sig['adjacent']).sum() if len(sig) > 0 else 0
        
        table = np.array([
            [adj_sig, adj_tested - adj_sig],
            [nonadj_sig, nonadj_tested - nonadj_sig]
        ])
        try:
            _, adj_pvalue = stats.fisher_exact(table)
        except ValueError:
            adj_pvalue = 1.0
        
        summary['adjacent_pairs_tested'] = int(adj_tested)
        summary['adjacent_pairs_significant'] = int(adj_sig)
        summary['adjacent_enrichment_pvalue'] = adj_pvalue
    
    return summary


def _phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    """
    Compute phi coefficient for a 2x2 contingency table.
    
         | B=0 | B=1
    A=0  |  a  |  b
    A=1  |  c  |  d
    
    phi = (a*d - b*c) / sqrt((a+b)(c+d)(a+c)(b+d))
    """
    numerator = a * d - b * c
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _benjamini_hochberg(pvalues: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return []
    
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    fdr = [0.0] * n
    
    cummin = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx, pval = indexed[rank_minus_1]
        rank = rank_minus_1 + 1
        adjusted = pval * n / rank
        cummin = min(cummin, adjusted)
        fdr[orig_idx] = min(cummin, 1.0)
    
    return fdr

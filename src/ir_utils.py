"""
ir_utils.py — Shared utilities for long-read IR analysis.

Handles:
- GTF parsing for intron coordinates
- BAM read processing for long-read data
- IR ratio computation compatible with IRFinder logic
"""

import bisect
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pysam


@dataclass
class Intron:
    """Represents an intron with its genomic coordinates and parent gene."""
    chrom: str
    start: int  # 0-based
    end: int    # 0-based, exclusive
    strand: str
    gene_id: str
    gene_name: str
    transcript_id: str
    intron_index: int  # index within the transcript (0-based)
    upstream_exon_end: int
    downstream_exon_start: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}:{self.strand}"

    @property
    def gene_intron_key(self) -> str:
        """Key for grouping introns by gene."""
        return f"{self.gene_id}:intron_{self.intron_index}"


@dataclass
class IREvent:
    """An intron retention measurement from a single sample."""
    intron: Intron
    intronic_reads: int       # reads fully within the intron
    splice_reads: int         # reads spliced across the intron
    ir_ratio: float           # intronic / (intronic + splice)
    coverage_fraction: float  # fraction of intron bases covered


def parse_introns_from_gtf(gtf_path: str,
                           gene_type: str = "protein_coding",
                           min_intron_length: int = 100,
                           max_intron_length: int = 500000) -> List[Intron]:
    """
    Extract all introns from a GTF file.
    
    Groups exons by transcript, sorts them, and derives introns as
    the gaps between consecutive exons.
    
    Args:
        gtf_path: Path to GENCODE GTF file.
        gene_type: Filter for this gene biotype.
        min_intron_length: Skip introns shorter than this.
        max_intron_length: Skip introns longer than this.
    
    Returns:
        List of Intron objects.
    """
    print(f"Parsing introns from {gtf_path}...")
    
    # First pass: collect exons per transcript
    transcripts = {}  # transcript_id -> {gene info + list of exon coords}
    
    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'exon':
                continue
            
            attrs = _parse_gtf_attributes(fields[8])
            
            # Filter for protein-coding genes
            if gene_type and attrs.get('gene_type', '') != gene_type:
                continue
            
            tx_id = attrs.get('transcript_id', '')
            if not tx_id:
                continue
            
            if tx_id not in transcripts:
                transcripts[tx_id] = {
                    'gene_id': attrs.get('gene_id', ''),
                    'gene_name': attrs.get('gene_name', ''),
                    'chrom': fields[0],
                    'strand': fields[6],
                    'exons': []
                }
            
            # GTF is 1-based inclusive; convert to 0-based half-open
            start = int(fields[3]) - 1
            end = int(fields[4])
            transcripts[tx_id]['exons'].append((start, end))
    
    print(f"  Found {len(transcripts)} protein-coding transcripts")
    
    # Second pass: derive introns from sorted exons
    introns = []
    seen_intron_keys = set()
    
    for tx_id, tx_info in transcripts.items():
        exons = sorted(tx_info['exons'], key=lambda x: x[0])
        
        if len(exons) < 2:
            continue
        
        for i in range(len(exons) - 1):
            intron_start = exons[i][1]      # end of upstream exon
            intron_end = exons[i + 1][0]    # start of downstream exon
            
            length = intron_end - intron_start
            if length < min_intron_length or length > max_intron_length:
                continue
            
            key = f"{tx_info['chrom']}:{intron_start}-{intron_end}:{tx_info['strand']}"
            if key in seen_intron_keys:
                continue
            seen_intron_keys.add(key)
            
            intron = Intron(
                chrom=tx_info['chrom'],
                start=intron_start,
                end=intron_end,
                strand=tx_info['strand'],
                gene_id=tx_info['gene_id'],
                gene_name=tx_info['gene_name'],
                transcript_id=tx_id,
                intron_index=i,
                upstream_exon_end=exons[i][1],
                downstream_exon_start=exons[i + 1][0],
            )
            introns.append(intron)
    
    print(f"  Extracted {len(introns)} unique introns from {len(transcripts)} transcripts")
    return introns


def _make_chrom_map(bam_references: Tuple, introns: List["Intron"]) -> Dict[str, Optional[str]]:
    """
    Build a mapping from GTF chromosome names to BAM reference names.

    Handles the common UCSC (chr1) vs ENSEMBL (1) mismatch by trying both
    the raw name and the chr-stripped / chr-prefixed variant.
    """
    bam_refs = set(bam_references)
    gtf_chroms = {i.chrom for i in introns}
    mapping: Dict[str, Optional[str]] = {}
    for chrom in gtf_chroms:
        if chrom in bam_refs:
            mapping[chrom] = chrom
        elif chrom.startswith('chr') and chrom[3:] in bam_refs:
            mapping[chrom] = chrom[3:]
        elif ('chr' + chrom) in bam_refs:
            mapping[chrom] = 'chr' + chrom
        else:
            mapping[chrom] = None
    matched = sum(1 for v in mapping.values() if v is not None)
    if matched == 0:
        print("  WARNING: no GTF chromosomes matched BAM references — check naming convention")
    else:
        print(f"  Chromosome map: {matched}/{len(gtf_chroms)} GTF contigs matched in BAM")
    return mapping


def compute_ir_from_bam(bam_path: str,
                         introns: List[Intron],
                         min_coverage: int = 3,
                         min_overhang: int = 10) -> List[IREvent]:
    """
    Compute IR ratios from a long-read BAM file.

    For each intron:
    - Count reads that are spliced across it (splice junction reads)
    - Count reads that span into the intron (retention reads)
    - Compute IR ratio = retention / (retention + spliced)

    Long-read specific: a single read can span the entire intron,
    so we check the CIGAR for 'N' operations (splice) vs 'M' (match)
    across the intron coordinates.

    Args:
        bam_path: Path to sorted, indexed BAM file.
        introns: List of Intron objects to test.
        min_coverage: Minimum total reads (splice + retention) to report.
        min_overhang: Minimum bases a read must extend past the intron
                      boundary on each side.

    Returns:
        List of IREvent objects.
    """
    bamfile = pysam.AlignmentFile(bam_path, "rb")
    chrom_map = _make_chrom_map(bamfile.references, introns)

    # Group introns by BAM chromosome, sorted by start for binary search
    chrom_introns: Dict[str, List[Intron]] = defaultdict(list)
    for intron in introns:
        bam_chrom = chrom_map.get(intron.chrom)
        if bam_chrom is not None:
            chrom_introns[bam_chrom].append(intron)
    for lst in chrom_introns.values():
        lst.sort(key=lambda x: x.start)

    events = []
    total_chroms = len(chrom_introns)

    for chrom_idx, (bam_chrom, chrom_int_list) in enumerate(chrom_introns.items()):
        print(f"  Scanning {bam_chrom} ({chrom_idx + 1}/{total_chroms},"
              f" {len(chrom_int_list)} introns)...")

        # Counters indexed by position in the sorted list
        n = len(chrom_int_list)
        splice_counts = [0] * n
        retention_counts = [0] * n
        intron_starts = [i.start for i in chrom_int_list]

        for read in bamfile.fetch(bam_chrom):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.cigartuples is None:
                continue

            read_start = read.reference_start
            read_end = read.reference_end  # may be None for reads with no aligned bases
            if read_end is None:
                continue

            # Binary-search: first intron whose start >= read_start - min_overhang
            lo = bisect.bisect_left(intron_starts, read_start - min_overhang)

            for i in range(lo, n):
                intron = chrom_int_list[i]
                # Once intron start exceeds read end we can stop
                if intron.start > read_end:
                    break
                # Read must fully span this intron (with overhang)
                if read_start > intron.start - min_overhang:
                    continue
                if read_end < intron.end + min_overhang:
                    continue

                classification = _classify_read_at_intron(read, intron.start, intron.end)
                if classification == 'spliced':
                    splice_counts[i] += 1
                elif classification == 'retained':
                    retention_counts[i] += 1

        for i, intron in enumerate(chrom_int_list):
            total = splice_counts[i] + retention_counts[i]
            if total < min_coverage:
                continue
            ir_ratio = retention_counts[i] / total
            events.append(IREvent(
                intron=intron,
                intronic_reads=retention_counts[i],
                splice_reads=splice_counts[i],
                ir_ratio=ir_ratio,
                coverage_fraction=1.0,
            ))

    bamfile.close()
    return events


def _classify_read_at_intron(read: pysam.AlignedSegment,
                              intron_start: int,
                              intron_end: int,
                              tolerance: int = 30) -> str:
    """
    Classify a read as 'spliced', 'retained', or 'ambiguous' at a given intron.

    Spliced if:
      (a) an N CIGAR op lands within `tolerance` bp of both intron boundaries, OR
      (b) an N CIGAR op overlaps ≥80% of the intron length (catches offset junctions
          common in noisy nanopore alignments).

    Retained if:
      continuous alignment covers >70% of the intron AND no N op overlaps the
      intron by more than 50 bp (a read with a splice inside the intron is
      ambiguous, not retained).

    Everything else is ambiguous.
    """
    intron_length = intron_end - intron_start
    ref_pos = read.reference_start
    has_matching_splice = False
    n_overlaps_intron = False   # any N with >50bp overlap of the intron
    intron_aligned_bases = 0
    large_gap_count = 0

    for op, length in read.cigartuples:
        if op in (0, 7, 8):  # M, =, X
            overlap_start = max(ref_pos, intron_start)
            overlap_end = min(ref_pos + length, intron_end)
            if overlap_end > overlap_start:
                intron_aligned_bases += overlap_end - overlap_start
            ref_pos += length
        elif op == 1:  # I — doesn't consume reference
            pass
        elif op == 2:  # D
            if length > 50 and ref_pos >= intron_start and ref_pos + length <= intron_end:
                large_gap_count += 1
            ref_pos += length
        elif op == 3:  # N (splice junction)
            splice_start = ref_pos
            splice_end = ref_pos + length
            # Rule (a): boundaries match within tolerance
            if (abs(splice_start - intron_start) <= tolerance and
                    abs(splice_end - intron_end) <= tolerance):
                has_matching_splice = True
            # Rule (b): N spans ≥80% of the intron
            if intron_length > 0:
                n_overlap = min(splice_end, intron_end) - max(splice_start, intron_start)
                if n_overlap > 0:
                    if n_overlap / intron_length >= 0.8:
                        has_matching_splice = True
                    if n_overlap > 50:
                        n_overlaps_intron = True
            ref_pos += length
        elif op in (4, 5):  # S, H
            pass
        else:
            ref_pos += length

    if has_matching_splice:
        return 'spliced'

    if large_gap_count > 2:
        return 'ambiguous'

    # Retained: >70% continuous alignment through intron, no internal splice signal
    if (intron_length > 0
            and intron_aligned_bases / intron_length > 0.7
            and not n_overlaps_intron):
        return 'retained'

    return 'ambiguous'


_IR_COLUMNS = [
    'chrom', 'intron_start', 'intron_end', 'strand', 'gene_id', 'gene_name',
    'transcript_id', 'intron_index', 'intron_length', 'intronic_reads',
    'splice_reads', 'ir_ratio', 'coverage_fraction',
]


def ir_events_to_dataframe(events: List[IREvent]) -> pd.DataFrame:
    """Convert list of IREvent to a pandas DataFrame."""
    if not events:
        return pd.DataFrame(columns=_IR_COLUMNS)
    rows = []
    for e in events:
        rows.append({
            'chrom': e.intron.chrom,
            'intron_start': e.intron.start,
            'intron_end': e.intron.end,
            'strand': e.intron.strand,
            'gene_id': e.intron.gene_id,
            'gene_name': e.intron.gene_name,
            'transcript_id': e.intron.transcript_id,
            'intron_index': e.intron.intron_index,
            'intron_length': e.intron.length,
            'intronic_reads': e.intronic_reads,
            'splice_reads': e.splice_reads,
            'ir_ratio': e.ir_ratio,
            'coverage_fraction': e.coverage_fraction,
        })
    return pd.DataFrame(rows)


def _parse_gtf_attributes(attr_string: str) -> Dict[str, str]:
    """Parse GTF attribute column into a dictionary."""
    attrs = {}
    for item in attr_string.strip().split(';'):
        item = item.strip()
        if not item:
            continue
        # Handle key "value" format
        match = re.match(r'(\S+)\s+"([^"]*)"', item)
        if match:
            attrs[match.group(1)] = match.group(2)
        else:
            # Handle key value format (no quotes)
            parts = item.split(None, 1)
            if len(parts) == 2:
                attrs[parts[0]] = parts[1].strip('"')
    return attrs

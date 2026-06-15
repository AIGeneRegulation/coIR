# Co-retention: single-molecule analysis of coordinated intron retention

Code accompanying:

**Coordinated intron retention on single RNA molecules marks fitness-critical genes**  
W. Ritchie, *Nature Communications* (submitted)

## Overview

This repository provides tools to detect and characterise coordinated intron retention (co-retention) from long-read RNA sequencing data. Co-retention occurs when adjacent introns are retained on the same transcript molecule, and can be distinguished from independent retention only with single-molecule sequencing.

The pipeline takes aligned nanopore direct RNA BAMs and produces:
- Per-intron retention calls from long reads
- Co-retention statistics for adjacent intron pairs (Fisher's exact test, phi coefficients)
- RBP motif enrichment at co-retained vs independently retained introns
- Integration with ENCODE eCLIP binding data
- Splice site strength and intervening exon length comparisons
- Cross-referencing with DepMap fitness scores, protein complex membership, Gene Ontology
- Splicing order integration (Choquet et al. 2023)

## Requirements

- Python 3.10+
- numpy, pandas, scipy, statsmodels
- pysam (for BAM parsing)
- pyfaidx (for genome sequence extraction)
- pyBigWig (for ENCODE signal tracks)
- matplotlib, seaborn (for figures)

Install:
```
pip install numpy pandas scipy statsmodels pysam pyfaidx pyBigWig matplotlib seaborn pybedtools pyranges tqdm
```

## Input data

- Nanopore direct RNA BAMs aligned to GRCh38 (e.g. from [SG-NEx](https://github.com/GoekeLab/sg-nex-data))
- GENCODE GTF annotation (v44 used in the paper)
- GRCh38 reference FASTA
- ENCODE eCLIP narrowPeak files (for binding validation)
- ENCODE ChIP-seq bigWigs (optional, for chromatin analysis)

## Usage

### 1. Intron retention detection and co-retention analysis

```bash
python src/run_pilot.py \
    --bam <aligned.bam> \
    --gtf <gencode.v44.annotation.gtf> \
    --outdir results/<sample>/ \
    --min-ir-ratio 0.05 \
    --min-coverage 5 \
    --min-coret-reads 10 \
    --max-pair-distance 5
```

Outputs:
- `*_ir_events.tsv` — per-intron retention ratios
- `*_coretention.tsv` — co-retention statistics for all tested pairs
- `*_summary.json` — summary statistics

### 2. RBP motif enrichment

```bash
python src/rbp_motifs.py \
    --coretention-tsv results/<sample>/*_coretention.tsv \
    --ir-tsv results/<sample>/*_ir_events.tsv \
    --genome-fasta <GRCh38.fa> \
    --outdir results/<sample>/motifs/
```

### 3. eCLIP binding validation

```bash
python src/eclip_validation.py \
    --coretention-tsv results/<sample>/*_coretention.tsv \
    --ir-tsv results/<sample>/*_ir_events.tsv \
    --eclip-dir <encode_eclip/> \
    --outdir results/<sample>/eclip/
```

### 4. Splice site and exon architecture

```bash
python src/splice_site_strength.py \
    --coretention-tsv results/<sample>/*_coretention.tsv \
    --ir-tsv results/<sample>/*_ir_events.tsv \
    --genome-fasta <GRCh38.fa> \
    --outdir results/<sample>/splice_sites/
```

### 5. Gene function analysis

```bash
python src/gene_function_analysis.py \
    --gene-list results/co_retained_genes.tsv \
    --depmap-file <CRISPRGeneEffect.csv> \
    --go-obo <go-basic.obo> \
    --go-gaf <goa_human.gaf.gz> \
    --gnomad-file <gnomad.v2.1.1.lof_metrics.by_gene.txt> \
    --corum-file <allComplexes.txt> \
    --outdir results/gene_function/
```

### 6. Expression-matched controls

```bash
python src/reviewer_controls.py \
    --coretention-tsv results/<sample>/*_coretention.tsv \
    --ir-tsv results/<sample>/*_ir_events.tsv \
    --depmap-file <CRISPRGeneEffect.csv> \
    --bam <aligned.bam> \
    --outdir results/controls/
```

## Data availability

- SG-NEx direct RNA sequencing: [AWS Open Data](https://registry.opendata.aws/sg-nex-data/)
- ENCODE eCLIP and ChIP-seq: [ENCODE portal](https://www.encodeproject.org)
- Splicing order data: [GEO GSE232455](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE232455)
- DepMap CRISPR fitness: [DepMap portal](https://depmap.org/portal/download/)

## Citation

If you use this code, please cite:

Ritchie, W. Coordinated intron retention on single RNA molecules marks fitness-critical genes. *Nat. Struct. Mol. Biol.* (submitted).

## Related software

- [IRFinder](https://github.com/RitchieLabIGH/IRFinder) — intron retention detection from short-read RNA-seq (Middleton et al., Genome Biology, 2017)

## License

MIT License

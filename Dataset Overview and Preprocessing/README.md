# Dataset & Preprocessing — eDNA Novel Taxonomy Discovery

> **Paper:** *Identifying Novel Taxonomy from eDNA Datasets using Multi-scale FCGR and Invariant Information Clustering*
> **Authors:** Sanskriti Tiwari, R. Lavanya, Iraa Jayakumar
> **Institution:** Department of Computing Technologies, SRM Institute of Science and Technology, Chennai, India

---

---

## 1. Dataset Overview

| Field              | Details                                                    |
|--------------------|------------------------------------------------------------|
| **Source**         | NCBI Sequence Read Archive (SRA)                          |
| **Accession**      | `SRX9086406`                                              |
| **BioProject**     | `PRJNA658834`                                             |
| **Sequencing**     | Illumina MiSeq                                            |
| **Marker gene**    | 18S rRNA (eukaryotic amplicon)                            |
| **Sample type**    | Deep-sea sediment — Mariana Trench                        |
| **Raw reads**      | 89,773 spots                                              |
| **Format**         | FASTQ (paired-end)                                        |

**Why this dataset?**
The Mariana Trench sediment eDNA dataset exhibits properties that make conventional reference-based pipelines inadequate: short read lengths, PCR-induced artifacts, chimera formation, and pronounced abundance sparsity between taxonomic groups. These characteristics make it ideal for benchmarking unsupervised, alignment-free learning frameworks.

---

## 2. Requirements

### Software

| Tool        | Version  | Purpose                          | Install                                   |
|-------------|----------|----------------------------------|-------------------------------------------|
| `SRA Toolkit` | ≥ 3.0   | Downloading SRA data             | https://github.com/ncbi/sra-tools         |
| `cutadapt`  | ≥ 4.x    | Adapter trimming, quality filter | `pip install cutadapt`                    |
| `VSEARCH`   | v2.x     | Filtering, derep, chimera removal| https://github.com/torognes/vsearch       |
| `seqkit`    | ≥ 2.x    | QC validation                    | https://bioinf.shenwei.me/seqkit/         |

### Verify installations

```bash
fastq-dump --version
cutadapt --version
vsearch --version
seqkit version
```

---

## 3. Preprocessing Pipeline Overview

```
Raw eDNA Reads (FASTQ)
        │
        ▼
Step 1: Download (SRA Toolkit)
        │
        ▼
Step 2: Adapter Removal & Quality Trimming  ──── Tool: cutadapt
        │                                         Removes non-biological adapters; Q20+ filter
        ▼
Step 3: Low-Quality Read Filtering  ─────────── Tool: VSEARCH
        │                                         Expected error model (maxEE ≤ 2)
        ▼
Step 4: Dereplication with Abundance Preservation  ── Tool: VSEARCH
        │                                              Collapses identical reads; retains counts
        ▼
Step 5: De Novo Chimera Detection & Removal  ──── Tool: VSEARCH (uchime_denovo)
        │                                          Removes PCR artifacts
        ▼
Step 6: QC Validation  ──────────────────────── Tool: seqkit
        │                                         Confirms 301 bp uniform length, GC 38–70%
        ▼
Clean High-Quality Sequences (FASTA)
→ Ready for FCGR Image Generation & ML Modelling
```

---

## 4. Step-by-Step Preprocessing Commands

### Step 1 — Download the Dataset

Use the SRA Toolkit to download the raw FASTQ reads.

```bash
# Download paired-end reads from SRA
fastq-dump --split-files --gzip SRX9086406

# Output files:
#   SRX9086406_1.fastq.gz   (forward reads)
#   SRX9086406_2.fastq.gz   (reverse reads)
```

> **Note:** Use `--split-files` to separate paired reads. The `--gzip` flag compresses output to save disk space.

Alternatively, using `fasterq-dump` for faster parallel download:

```bash
fasterq-dump --split-files SRX9086406
gzip SRX9086406_1.fastq SRX9086406_2.fastq
```

---

### Step 2 — Adapter Removal & Base-Quality Trimming

Remove Illumina adapter sequences and filter low-confidence bases using `cutadapt`.

```bash
cutadapt \
  -a AGATCGGAAGAGCACACGTCTGAAC \
  -A AGATCGGAAGAGCGTCGTGTAGGGA \
  -q 20,20 \
  --minimum-length 100 \
  -o trimmed_R1.fastq.gz \
  -p trimmed_R2.fastq.gz \
  SRX9086406_1.fastq.gz \
  SRX9086406_2.fastq.gz \
  > cutadapt_report.txt 2>&1
```

**Parameter explanation:**

| Parameter            | Value                      | Purpose                                                   |
|----------------------|----------------------------|-----------------------------------------------------------|
| `-a`                 | Illumina adapter (R1)      | Trims 3′ adapter from forward reads                      |
| `-A`                 | Illumina adapter (R2)      | Trims 3′ adapter from reverse reads                      |
| `-q 20,20`           | Quality score threshold    | Trims bases with Phred score < 20 from both ends         |
| `--minimum-length`   | `100`                      | Discards reads shorter than 100 bp after trimming        |
| `-o / -p`            | Output files               | Trimmed forward and reverse reads                        |

**Expected output:** ~170,404 adapter-free, Q20+ reads (approx. +90% from raw spots after paired expansion).

---

### Step 3 — Low-Quality Read Filtering

Apply an expected error (EE) model to remove reads dominated by sequencing errors while retaining short but biologically valid fragments common in deep-sea eDNA.

```bash
vsearch \
  --fastq_filter trimmed_R1.fastq.gz \
  --fastq_maxee 2 \
  --fastq_trunclen 301 \
  --fastaout filtered.fasta \
  --log filter_log.txt
```

**Parameter explanation:**

| Parameter          | Value   | Purpose                                                            |
|--------------------|---------|--------------------------------------------------------------------|
| `--fastq_maxee`    | `2`     | Maximum expected errors per read; discards reads exceeding this   |
| `--fastq_trunclen` | `301`   | Truncates all reads to uniform length of 301 bp                   |
| `--fastaout`       | path    | Outputs passing reads as FASTA                                     |

> **Key decision:** `maxEE = 2` balances stringent error removal with preservation of rare, low-abundance taxa that carry genuine biological signals.

**Expected output:** 170,404 reads retained (0% reduction at this step — quality already ensured by Step 2).

---

### Step 4 — Dereplication with Abundance Preservation

Collapse identical sequences to reduce computational redundancy. Critically, abundance information (read counts) is retained as size annotations for use in downstream abundance-aware processing.

```bash
vsearch \
  --fastx_uniques filtered.fasta \
  --sizeout \
  --relabel uniq \
  --fasta_width 0 \
  --output dereplicated.fasta \
  --uc derep_clusters.uc \
  --log derep_log.txt
```

**Parameter explanation:**

| Parameter        | Value  | Purpose                                                          |
|------------------|--------|------------------------------------------------------------------|
| `--fastx_uniques`| path   | Input FASTA; identifies unique sequences                        |
| `--sizeout`      | flag   | Appends `;size=N` abundance annotation to each sequence header  |
| `--relabel`      | `uniq` | Relabels sequences with a clean prefix                          |
| `--fasta_width`  | `0`    | Disables line-wrapping in output FASTA                          |
| `--uc`           | path   | Outputs cluster information file                                |

> **Important:** The `--sizeout` flag is essential. These abundance annotations drive the abundance-aware mimic generation and weighted clustering in later pipeline stages.

**Expected output:** 59,200 unique sequences (65.2% reduction from 170,404).

---

### Step 5 — De Novo Chimera Detection & Removal

Detect and remove PCR-induced chimeric sequences using the `uchime_denovo` algorithm. De novo chimera detection operates without a reference database, making it suitable for deep-sea samples with limited database coverage.

```bash
vsearch \
  --uchime_denovo dereplicated.fasta \
  --sizein \
  --nonchimeras non_chimeric.fasta \
  --chimeras chimeric.fasta \
  --uchimeout chimera_report.txt \
  --log chimera_log.txt
```

**Parameter explanation:**

| Parameter          | Value | Purpose                                              |
|--------------------|-------|------------------------------------------------------|
| `--uchime_denovo`  | path  | Input dereplicated FASTA; performs de novo detection |
| `--sizein`         | flag  | Reads size annotations added in Step 4              |
| `--nonchimeras`    | path  | Output: clean, chimera-free sequences               |
| `--chimeras`       | path  | Output: flagged chimeric sequences (for inspection) |
| `--uchimeout`      | path  | Detailed chimera scoring report                     |

> **Note on over-filtering:** Excessive chimera filtering can eliminate genuine rare variants in poorly characterized environments like the deep sea. The de novo approach is calibrated to VSEARCH defaults to minimize false positives while removing clear PCR artifacts.

**Expected output:** 57,221 non-chimeric sequences (2.8% reduction from 59,200).

---

### Step 6 — QC Validation

Validate the final dataset to confirm uniform read length, acceptable GC content range, and absence of artifacts.

```bash
# Check sequence length distribution
seqkit stats -a non_chimeric.fasta > final_stats.txt

# Confirm all reads are exactly 301 bp
seqkit seq --min-len 301 --max-len 301 non_chimeric.fasta | seqkit stats

# Check GC content distribution
seqkit fx2tab --gc non_chimeric.fasta | \
  awk '{print $3}' | sort -n > gc_content_values.txt

# View summary
cat final_stats.txt
```

**Validation checklist:**

| Check                     | Expected Value         | Significance                                     |
|---------------------------|------------------------|--------------------------------------------------|
| Total sequences           | 57,221                 | Final unique, high-quality sequences             |
| Read length               | 301 bp (uniform)       | Confirms consistent trimming                     |
| GC content range          | 38% – 70%              | Multimodal → diverse taxa, not sequencing bias   |
| Chimeric sequences        | 0%                     | No PCR artifacts in final output                 |
| Low-complexity sequences  | 0%                     | No homopolymer or dust-masked reads              |

```bash
# Optional: Generate a detailed per-sequence summary table
seqkit fx2tab --length --gc --name non_chimeric.fasta > per_seq_summary.tsv
```

**Final validated output file:** `non_chimeric.fasta`
This file is directly used as input for FCGR image construction and downstream unsupervised machine learning.

---

## 5. Expected Output at Each Stage

| Stage | Tool | Input Seqs | Output Seqs | Reduction | Key Quality Improvement |
|---|---|---|---|---|---|
| Raw MiSeq | — | 89,773 spots | — | — | — |
| 1. Adapter/Quality | cutadapt | 89,773 | 170,404 | +90% | Adapter-free, Q20+ |
| 2. Error Filtering | vsearch | 170,404 | 170,404 | 0% | maxEE ≤ 2 validated |
| 3. Dereplication | vsearch | 170,404 | 59,200 | 65.2% | Abundance preserved |
| 4. Chimera Removal | vsearch | 59,200 | 57,221 | 2.8% | 0% PCR artifacts |
| 5. QC Validation | seqkit | 57,221 | 57,221 | 0% | 301 bp uniform, GC 38–70% |

---

## 6. Final Dataset Characteristics

After preprocessing, the final dataset has the following biological properties:

- **57,221 unique, non-redundant, chimera-free sequences**
- **Uniform read length:** 301 bp — ensures consistent FCGR matrix dimensions across all sequences
- **GC content:** 38%–70% with a multimodal distribution, reflecting the presence of diverse taxonomic groups rather than sequencing bias
- **Abundance distribution:** Long-tailed with a median abundance of 1 and a maximum of 7,421 — consistent with ecological biodiversity profiles where rare taxa dominate numerically
- **No PCR chimeras or low-complexity sequences** detected in the final output

These properties confirm that the preprocessing pipeline effectively removed sequencing noise while preserving biologically meaningful diversity, enabling reliable FCGR image generation and downstream machine learning.

---

## 7. Directory Structure

```
project/
│
├── data/
│   ├── raw/
│   │   ├── SRX9086406_1.fastq.gz          # Raw forward reads (SRA)
│   │   └── SRX9086406_2.fastq.gz          # Raw reverse reads (SRA)
│   │
│   ├── processed/
│   │   ├── trimmed_R1.fastq.gz            # After Step 2: adapter trimmed
│   │   ├── trimmed_R2.fastq.gz            # After Step 2: adapter trimmed
│   │   ├── filtered.fasta                 # After Step 3: error filtered
│   │   ├── dereplicated.fasta             # After Step 4: dereplicated
│   │   ├── derep_clusters.uc              # Dereplication cluster map
│   │   ├── non_chimeric.fasta             # After Step 5: final clean sequences ✅
│   │   └── chimeric.fasta                 # Flagged chimeras (for inspection)
│   │
│   └── qc/
│       ├── cutadapt_report.txt            # Adapter trimming summary
│       ├── filter_log.txt                 # Error filtering log
│       ├── derep_log.txt                  # Dereplication log
│       ├── chimera_report.txt             # Chimera scoring report
│       ├── final_stats.txt                # seqkit QC summary
│       ├── gc_content_values.txt          # Per-sequence GC values
│       └── per_seq_summary.tsv            # Full per-sequence table
│
└── README_Dataset_Preprocessing.md
```

---


#!/usr/bin/env python3

import sys
import argparse
import subprocess
import logging
import bisect
from collections import defaultdict
from pathlib import Path

import pysam
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from Bio import SeqIO

def parse_args():
    p = argparse.ArgumentParser(description="dRNA-seq TSS detection and promoter extraction")
    p.add_argument('--tex_plus', required=True, help="Raw TEX(+) FASTQ file")
    p.add_argument('--tex_minus', required=True, help="Raw TEX(-) FASTQ file")
    p.add_argument('--fasta', required=True, help="Reference genome FASTA")
    p.add_argument('--gff3', required=True, help="Genome annotation GFF3")
    p.add_argument('--out_dir', required=True, help="Output directory")
    p.add_argument('--threads', type=int, default=8, help="CPU threads to use")

    p.add_argument('--min_read_len', type=int, default=20, help="Minimum read length after trimming")
    p.add_argument('--strandedness', choices=['auto', 'forward', 'reverse'], default='auto',
                   help="Library strandedness")

    p.add_argument('--min_raw_plus', type=int, default=5, help="Minimum raw reads at TSS in TEX(+)")
    p.add_argument('--fc_thresh', type=float, default=2.0, help="Minimum normalized TEX(+)/TEX(-) fold change")
    p.add_argument('--step_factor', type=float, default=2.0, help="Minimum ratio of TSS signal vs upstream (-1 to -3 bp)")
    p.add_argument('--p_cutoff', type=float, default=0.01, help="FDR adjusted p-value cutoff")
    p.add_argument('--peak_mindist', type=int, default=5, help="Minimum distance between adjacent TSSs (bp)")

    p.add_argument('--promoter_up', type=int, default=99, help="Upstream bases from TSS")
    p.add_argument('--promoter_down', type=int, default=0, help="Downstream bases from TSS")
    return p.parse_args()

def setup_logger(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(out_dir) / "pipeline.log"

    logger = logging.getLogger("tss")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def run_cmd(cmd, logger):
    logger.info(f"Running: {cmd}")
    full_cmd = f"bash -c 'set -o pipefail; {cmd}'"
    result = subprocess.run(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
        sys.exit(1)
    return result

def run_fastp(in_fq, out_fq, sample_name, out_dir, threads, min_len, logger):
    html_report = Path(out_dir) / f"{sample_name}_fastp.html"
    json_report = Path(out_dir) / f"{sample_name}_fastp.json"

    cmd = (
        f"fastp -i {in_fq} -o {out_fq} "
        f"--thread {threads} "
        f"--trim_poly_x "
        f"--cut_tail "
        f"--cut_mean_quality 20 "
        f"--qualified_quality_phred 20 "
        f"--unqualified_percent_limit 20 "
        f"--length_required {min_len} "
        f"-h {html_report} -j {json_report}"
    )
    logger.info(f"fastp: {sample_name}")
    run_cmd(cmd, logger)

def build_bowtie2_index(fasta, out_dir, threads, logger):
    prefix = Path(out_dir) / "bt2_idx" / "reference"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if Path(f"{prefix}.1.bt2").exists():
        logger.info("Using existing Bowtie2 index")
        return str(prefix)

    logger.info("Building Bowtie2 index")
    run_cmd(f"bowtie2-build --threads {threads} {fasta} {prefix}", logger)
    return str(prefix)

def align_bowtie2(trimmed_fq, idx_prefix, out_bam, threads, logger):
    stats_out = f"{out_bam}.align_stats.txt"
    cmd = (
        f"bowtie2 -p {threads} --end-to-end --very-sensitive -x {idx_prefix} -U {trimmed_fq} 2> {stats_out} "
        f"| samtools view -b -F 4 -q 20 - "
        f"| samtools sort -@ {threads} -o {out_bam} -"
    )
    logger.info(f"Aligning {trimmed_fq}")
    run_cmd(cmd, logger)
    run_cmd(f"samtools index {out_bam}", logger)

def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_int = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_int[0])]
    for current in sorted_int[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(list(current))
    return [tuple(x) for x in merged]

def load_intervals(gff3_fp):
    cds_raw = defaultdict(list)
    rna_mask_raw = defaultdict(list)

    with open(gff3_fp) as f:
        for line in f:
            if line.startswith('##FASTA'):
                break
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            chrom, _, feature, start, end, _, strand, _, _ = parts
            start_0, end_0 = int(start) - 1, int(end)
            feature_lower = feature.lower()

            if 'rna' in feature_lower:
                rna_mask_raw[chrom].append((start_0, end_0))
            elif feature_lower in ['cds', 'gene']:
                cds_raw[chrom].append((start_0, end_0, strand))

    rna_mask_merged = {ch: merge_intervals(intervals) for ch, intervals in rna_mask_raw.items()}

    for ch in cds_raw:
        cds_raw[ch].sort(key=lambda x: x[0])

    return cds_raw, rna_mask_merged

def is_in_mask(pos, mask_list):
    if not mask_list:
        return False
    idx = bisect.bisect_right(mask_list, (pos, float('inf'))) - 1
    if idx >= 0:
        start, end = mask_list[idx]
        if start <= pos < end:
            return True
    return False

def infer_strandedness(bam_fp, cds_intervals, logger):
    counts = {'forward': 0, 'reverse': 0}
    bam = pysam.AlignmentFile(bam_fp, 'rb')

    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.mapping_quality < 20:
            continue

        chrom = bam.get_reference_name(read.reference_id)
        if chrom not in cds_intervals:
            continue

        pos = read.reference_start
        feats = cds_intervals[chrom]

        starts = [f[0] for f in feats]
        idx = bisect.bisect_right(starts, pos) - 1

        if idx >= 0:
            start, end, strand = feats[idx]
            if start <= pos < end:
                read_is_rev = read.is_reverse
                if (not read_is_rev and strand == '+') or (read_is_rev and strand == '-'):
                    counts['forward'] += 1
                else:
                    counts['reverse'] += 1

        if (counts['forward'] + counts['reverse']) >= 10000:
            break

    bam.close()
    inferred = 'forward' if counts['forward'] >= counts['reverse'] else 'reverse'
    logger.info(f"Strandedness: forward={counts['forward']} reverse={counts['reverse']}")
    logger.info(f"Library orientation: {inferred}")
    return inferred

def extract_5prime_coverage(bam_fp, fasta_fp, rna_mask, lib_strand):
    fasta_lens = {rec.id: len(rec.seq) for rec in SeqIO.parse(fasta_fp, 'fasta')}
    cov = {
        '+': {ch: np.zeros(length, dtype=np.int32) for ch, length in fasta_lens.items()},
        '-': {ch: np.zeros(length, dtype=np.int32) for ch, length in fasta_lens.items()}
    }

    bam = pysam.AlignmentFile(bam_fp, 'rb')
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_qcfail or read.is_duplicate:
            continue

        chrom = bam.get_reference_name(read.reference_id)
        if chrom not in fasta_lens:
            continue

        if lib_strand == 'forward':
            if not read.is_reverse:
                tx_strand, pos = '+', read.reference_start
            else:
                tx_strand, pos = '-', read.reference_end - 1
        else:
            if not read.is_reverse:
                tx_strand, pos = '-', read.reference_start
            else:
                tx_strand, pos = '+', read.reference_end - 1

        mask_list = rna_mask.get(chrom, [])
        if is_in_mask(pos, mask_list):
            continue

        if 0 <= pos < fasta_lens[chrom]:
            cov[tx_strand][chrom][pos] += 1

    bam.close()
    return cov

def detect_tss(cov_p, cov_m, args, logger):
    lib_p = sum(np.sum(arr) for strand_dict in cov_p.values() for arr in strand_dict.values())
    lib_m = sum(np.sum(arr) for strand_dict in cov_m.values() for arr in strand_dict.values())
    logger.info(f"Library size (5' ends, rRNA/tRNA masked): TEX+ {lib_p}, TEX- {lib_m}")

    prob_null = lib_p / (lib_p + lib_m) if (lib_p + lib_m) > 0 else 0.5
    nf_p = 1e6 / lib_p if lib_p > 0 else 1.0
    nf_m = 1e6 / lib_m if lib_m > 0 else 1.0

    candidates = []

    for strand in ['+', '-']:
        for chrom in cov_p[strand]:
            cp = cov_p[strand][chrom]
            cm = cov_m[strand][chrom]
            c_len = len(cp)

            peaks, _ = find_peaks(cp, height=args.min_raw_plus, distance=args.peak_mindist)

            for p in peaks:
                val_plus = int(cp[p])
                val_minus = int(cm[p])

                if strand == '+':
                    up_coords = range(max(0, p - 3), p)
                else:
                    up_coords = range(p + 1, min(c_len, p + 4))

                up_mean = np.mean([cp[u] for u in up_coords]) if up_coords else 0.0
                step = val_plus / (up_mean + 1.0)
                if step < args.step_factor:
                    continue

                rpm_p = val_plus * nf_p
                rpm_m = val_minus * nf_m
                fc = (rpm_p + 1.0) / (rpm_m + 1.0)
                if fc < args.fc_thresh:
                    continue

                candidates.append({
                    'chr': chrom,
                    'pos_0based': p,
                    'pos_1based': p + 1,
                    'strand': strand,
                    'raw_plus': val_plus,
                    'raw_minus': val_minus,
                    'step_factor': round(step, 2),
                    'fc': round(fc, 2),
                    'total': val_plus + val_minus
                })

    if not candidates:
        logger.warning("No candidates passed the step-factor and fold-change filters")
        return []

    logger.info(f"Binomial test on {len(candidates)} candidates")

    pvals = []
    for c in candidates:
        res = binomtest(c['raw_plus'], c['total'], prob_null, alternative='greater')
        pvals.append(res.pvalue)
        c['pval'] = res.pvalue

    reject, qvals, _, _ = multipletests(pvals, alpha=args.p_cutoff, method='fdr_bh')

    significant_tss = []
    for c, sig, q in zip(candidates, reject, qvals):
        if sig:
            c['fdr_qval'] = q
            significant_tss.append(c)

    logger.info(f"{len(significant_tss)} TSSs at FDR < {args.p_cutoff}")
    return significant_tss

def extract_promoters(tss_list, fasta_fp, up, down):
    fasta = SeqIO.to_dict(SeqIO.parse(fasta_fp, 'fasta'))
    results = []

    for t in tss_list:
        ch = t['chr']
        pos = t['pos_0based']
        strand = t['strand']
        rec = fasta[ch]
        seq_len = len(rec)

        if strand == '+':
            start = max(0, pos - up)
            end = min(seq_len, pos + down + 1)
            seq = str(rec.seq[start:end])
        else:
            start = max(0, pos - down)
            end = min(seq_len, pos + up + 1)
            seq = str(rec.seq[start:end].reverse_complement())

        t_out = dict(t)
        t_out.pop('pos_0based', None)
        t_out['promoter_seq'] = seq
        t_out['seq_length'] = len(seq)
        results.append(t_out)

    return results

def main():
    args = parse_args()
    logger = setup_logger(args.out_dir)
    logger.info("Start")

    trimmed_plus = Path(args.out_dir) / "trimmed_tex_plus.fq.gz"
    trimmed_minus = Path(args.out_dir) / "trimmed_tex_minus.fq.gz"

    run_fastp(args.tex_plus, str(trimmed_plus), "TEX_plus", args.out_dir, args.threads, args.min_read_len, logger)
    run_fastp(args.tex_minus, str(trimmed_minus), "TEX_minus", args.out_dir, args.threads, args.min_read_len, logger)

    idx_prefix = build_bowtie2_index(args.fasta, args.out_dir, args.threads, logger)
    bam_plus = Path(args.out_dir) / "tex_plus.bam"
    bam_minus = Path(args.out_dir) / "tex_minus.bam"

    align_bowtie2(str(trimmed_plus), idx_prefix, str(bam_plus), args.threads, logger)
    align_bowtie2(str(trimmed_minus), idx_prefix, str(bam_minus), args.threads, logger)

    logger.info("Loading GFF3")
    cds_intervals, rna_mask_merged = load_intervals(args.gff3)

    if args.strandedness == 'auto':
        lib_strand = infer_strandedness(str(bam_plus), cds_intervals, logger)
    else:
        lib_strand = args.strandedness
        logger.info(f"Strandedness: {lib_strand}")

    logger.info("Counting 5' ends")
    cov_plus = extract_5prime_coverage(str(bam_plus), args.fasta, rna_mask_merged, lib_strand)
    cov_minus = extract_5prime_coverage(str(bam_minus), args.fasta, rna_mask_merged, lib_strand)

    tss_records = detect_tss(cov_plus, cov_minus, args, logger)

    if tss_records:
        logger.info(f"Extracting promoters (-{args.promoter_up} to +{args.promoter_down})")
        promoter_data = extract_promoters(tss_records, args.fasta, args.promoter_up, args.promoter_down)
        out_df = pd.DataFrame(promoter_data)
        out_tsv = Path(args.out_dir) / "TSS_promoters.tsv"
        out_df.to_csv(out_tsv, sep='\t', index=False)
        logger.info(f"Wrote {len(promoter_data)} promoters to {out_tsv}")
    else:
        logger.warning("No TSS passed filtering")

    logger.info("Done")

if __name__ == '__main__':
    main()

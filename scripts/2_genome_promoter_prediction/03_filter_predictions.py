import glob
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

WINDOW_LEN = 100
STEP_BP = 10


def read_fasta_lengths(fasta_path):
    lengths = {}
    name, length = None, 0
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = length
                name = line[1:].split()[0]
                length = 0
            else:
                length += len(line.strip())
        if name is not None:
            lengths[name] = length
    return lengths


def parse_gff(gff_path):
    cds_rows, rna_rows, gene_rows = [], [], []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid, source, ftype, start, end, score, strand, phase, attrs = parts
            attr = dict(kv.split("=", 1) for kv in attrs.split(";") if "=" in kv)

            if ftype == "CDS":
                cds_rows.append({"seqid": seqid, "start": int(start), "end": int(end),
                                  "strand": strand, "locus_tag": attr.get("locus_tag", "")})
            elif ftype == "gene":
                row = {"seqid": seqid, "start": int(start), "end": int(end),
                       "strand": strand, "locus_tag": attr.get("locus_tag", ""),
                       "name": attr.get("Name", attr.get("gene", "")),
                       "biotype": attr.get("gene_biotype", "")}
                gene_rows.append(row)
                if row["biotype"] in ("rRNA", "tRNA"):
                    rna_rows.append(row)
    return pd.DataFrame(cds_rows), pd.DataFrame(rna_rows), pd.DataFrame(gene_rows)


def _find_gff(accession_dir):
    preferred = os.path.join(accession_dir, "genomic.gff")
    if os.path.exists(preferred):
        return preferred
    candidates = glob.glob(os.path.join(accession_dir, "*.gff"))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        raise FileNotFoundError(f"No .gff file found in {accession_dir}")
    else:
        raise FileNotFoundError(
            f"Multiple .gff files in {accession_dir}, ambiguous: {candidates}")


def load_accession(accession_dir, accession):
    gff_path = _find_gff(accession_dir)
    fna_matches = glob.glob(os.path.join(accession_dir, f"{accession}_*.fna"))
    if not fna_matches:
        raise FileNotFoundError(f"No fasta matching {accession}_*.fna in {accession_dir}")
    scaffold_lengths = read_fasta_lengths(fna_matches[0])
    cds_df, rna_df, gene_df = parse_gff(gff_path)
    return scaffold_lengths, cds_df, rna_df, gene_df


def file_coord_to_genome(window_start_file, scaffold_len, strand,
                          window_len=WINDOW_LEN):
    s = window_start_file
    if strand == "+":
        return s, s + window_len - 1, s + window_len
    elif strand == "-":
        # reverse files are indexed from 1 at the last base of the scaffold
        L = scaffold_len
        genome_start = L - s - window_len + 2
        genome_end = L - s + 1
        tss = L - s - window_len + 1
        return genome_start, genome_end, tss
    else:
        raise ValueError("strand must be '+' or '-'")


def find_promoter_peaks(df, threshold=0.5, min_distance_bp=150,
                         prominence=0.1, step_bp=STEP_BP):
    df = df.sort_values("start_in_scaffold").reset_index(drop=True)
    probs = df["promoter_probability"].to_numpy()
    distance_steps = max(1, int(round(min_distance_bp / step_bp)))
    peak_idx, _ = find_peaks(probs, height=threshold,
                              distance=distance_steps, prominence=prominence)
    peaks = df.iloc[peak_idx].copy()
    peaks["peak_probability"] = probs[peak_idx]
    return peaks


def annotate_peak(genome_start, genome_end, tss, strand,
                   cds_df, rna_df, gene_df, max_upstream_dist=100):
    cds_hit = cds_df[(cds_df["strand"] == strand) &
                      (cds_df["start"] <= genome_end) & (cds_df["end"] >= genome_start)]
    rna_hit = rna_df[(rna_df["strand"] == strand) &
                      (rna_df["start"] <= genome_end) & (rna_df["end"] >= genome_start)]

    if strand == "+":
        downstream = gene_df[(gene_df["strand"] == "+") & (gene_df["start"] >= tss)]
        nearest = downstream.loc[downstream["start"].idxmin()] if len(downstream) else None
        dist = int(nearest["start"] - tss) if nearest is not None else None
    else:
        downstream = gene_df[(gene_df["strand"] == "-") & (gene_df["end"] <= tss)]
        nearest = downstream.loc[downstream["end"].idxmax()] if len(downstream) else None
        dist = int(tss - nearest["end"]) if nearest is not None else None

    if len(rna_hit):
        tier = "internal_rRNA_tRNA"
    elif len(cds_hit):
        tier = "internal_CDS"
    elif dist is not None and 0 <= dist <= max_upstream_dist:
        tier = "canonical"
    else:
        tier = "orphan"

    return {
        "tier": tier,
        "overlapping_cds": ",".join(cds_hit["locus_tag"].tolist()),
        "overlapping_rna": ",".join(rna_hit["locus_tag"].tolist()),
        "nearest_gene_locus_tag": nearest["locus_tag"] if nearest is not None else "",
        "nearest_gene_name": nearest["name"] if nearest is not None else "",
        "dist_to_downstream_gene": dist,
    }


def process_predictions(csv_path, accession_dir, accession, strand,
                         threshold=0.5, min_distance_bp=150, prominence=0.1,
                         max_upstream_dist=100):
    scaffold_lengths, cds_df, rna_df, gene_df = load_accession(accession_dir, accession)
    preds = pd.read_csv(csv_path, encoding="utf-8-sig")

    out = []
    for scaffold, sub in preds.groupby("scaffold"):
        if scaffold not in scaffold_lengths:
            raise KeyError(f"Scaffold '{scaffold}' not found in fasta headers "
                            f"for {accession} -- check name matching.")
        L = scaffold_lengths[scaffold]

        peaks = find_promoter_peaks(sub, threshold=threshold,
                                     min_distance_bp=min_distance_bp,
                                     prominence=prominence)

        cds_sub = cds_df[cds_df["seqid"] == scaffold]
        rna_sub = rna_df[rna_df["seqid"] == scaffold]
        gene_sub = gene_df[gene_df["seqid"] == scaffold]

        for _, p in peaks.iterrows():
            g_start, g_end, tss = file_coord_to_genome(
                int(p["start_in_scaffold"]), L, strand)
            ann = annotate_peak(g_start, g_end, tss, strand,
                                 cds_sub, rna_sub, gene_sub, max_upstream_dist)
            out.append({
                "accession": accession,
                "scaffold": scaffold,
                "strand": strand,
                "file_window_start": int(p["start_in_scaffold"]),
                "genome_window_start": g_start,
                "genome_window_end": g_end,
                "predicted_tss_genome": tss,
                "peak_probability": float(p["peak_probability"]),
                **ann,
            })

    return pd.DataFrame(out)


def run_accession(accession_dir, accession, forward_csv=None, reverse_csv=None, **kwargs):
    frames = []
    if forward_csv is not None:
        frames.append(process_predictions(forward_csv, accession_dir, accession, "+", **kwargs))
    if reverse_csv is not None:
        frames.append(process_predictions(reverse_csv, accession_dir, accession, "-", **kwargs))
    if not frames:
        raise ValueError("Provide at least one of forward_csv or reverse_csv")
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["scaffold", "genome_window_start"]).reset_index(drop=True)


def select_final_promoter_rows(csv_path, accession_dir, accession, strand,
                                threshold=0.5, min_distance_bp=150, prominence=0.1,
                                max_upstream_dist=100):
    file_strand = {"plus": "+", "minus": "-"}.get(strand)
    if file_strand is None:
        raise ValueError("strand must be 'plus' or 'minus'")

    scaffold_lengths, cds_df, rna_df, gene_df = load_accession(accession_dir, accession)
    preds = pd.read_csv(csv_path, encoding="utf-8-sig")

    keep_index = []
    for scaffold, sub in preds.groupby("scaffold"):
        if scaffold not in scaffold_lengths:
            raise KeyError(f"Scaffold '{scaffold}' not found in fasta headers "
                            f"for {accession} -- check name matching.")
        L = scaffold_lengths[scaffold]

        start_to_idx = dict(zip(sub["start_in_scaffold"], sub.index))

        peaks = find_promoter_peaks(sub, threshold=threshold,
                                     min_distance_bp=min_distance_bp,
                                     prominence=prominence)

        cds_sub = cds_df[cds_df["seqid"] == scaffold]
        rna_sub = rna_df[rna_df["seqid"] == scaffold]
        gene_sub = gene_df[gene_df["seqid"] == scaffold]

        for _, p in peaks.iterrows():
            file_start = int(p["start_in_scaffold"])
            g_start, g_end, tss = file_coord_to_genome(file_start, L, file_strand)
            ann = annotate_peak(g_start, g_end, tss, file_strand, cds_sub, rna_sub, gene_sub,
                                 max_upstream_dist=max_upstream_dist)
            if ann["tier"] == "canonical":
                keep_index.append(start_to_idx[file_start])

    result = preds.copy()
    result["promoter_probability"] = 0
    result.loc[keep_index, "promoter_probability"] = 1
    return result


def discover_prediction_files(predictions_dir, strand):
    REVERSE_SUFFIX = "_sequences_reverse_result.csv"
    FORWARD_SUFFIX = "_sequences_result.csv"
    if strand not in ("plus", "minus"):
        raise ValueError("strand must be 'plus' or 'minus'")

    found = []
    for fname in sorted(os.listdir(predictions_dir)):
        if strand == "minus":
            if fname.endswith(REVERSE_SUFFIX):
                accession = fname[: -len(REVERSE_SUFFIX)]
                found.append((accession, os.path.join(predictions_dir, fname)))
        else:
            if fname.endswith(FORWARD_SUFFIX) and not fname.endswith(REVERSE_SUFFIX):
                accession = fname[: -len(FORWARD_SUFFIX)]
                found.append((accession, os.path.join(predictions_dir, fname)))
    return found


def _filter_one_file(csv_path, genome_dir, output_dir, accession, strand,
                      threshold=0.5, min_distance_bp=150, prominence=0.1,
                      max_upstream_dist=100):
    accession_dir = os.path.join(genome_dir, accession)

    result = select_final_promoter_rows(csv_path, accession_dir, accession, strand,
                                         threshold=threshold, min_distance_bp=min_distance_bp,
                                         prominence=prominence, max_upstream_dist=max_upstream_dist)

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_filtered.csv")
    result.to_csv(out_path, index=False)
    return out_path


def filter_and_merge_to_file(genome_dir, predictions_dir, output_dir, accession, strand,
                              threshold=0.5, min_distance_bp=150, prominence=0.1,
                              max_upstream_dist=100):
    csv_name = (f"{accession}_sequences_result.csv" if strand == "plus"
                else f"{accession}_sequences_reverse_result.csv")
    csv_path = os.path.join(predictions_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    return _filter_one_file(csv_path, genome_dir, output_dir, accession, strand,
                             threshold=threshold, min_distance_bp=min_distance_bp,
                             prominence=prominence, max_upstream_dist=max_upstream_dist)


def filter_and_merge_all(genome_dir, predictions_dir, output_dir, strand,
                          threshold=0.5, min_distance_bp=150, prominence=0.1,
                          max_upstream_dist=100):
    files = discover_prediction_files(predictions_dir, strand)
    results = []
    for accession, csv_path in files:
        entry = {"accession": accession, "strand": strand, "csv_path": csv_path}
        try:
            out_path = _filter_one_file(csv_path, genome_dir, output_dir, accession, strand,
                                         threshold=threshold, min_distance_bp=min_distance_bp,
                                         prominence=prominence, max_upstream_dist=max_upstream_dist)
            entry["status"] = "ok"
            entry["out_path"] = out_path
        except Exception as e:
            entry["status"] = "error"
            entry["error_message"] = f"{type(e).__name__}: {e}"
        results.append(entry)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Peak calling and GFF filtering of promoter predictions")
    parser.add_argument("--genome_dir", required=True, help="Folder with one subfolder per accession (genomic.gff, *.fna)")
    parser.add_argument("--predictions_dir", required=True, help="Prediction CSVs of one strand")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--strand", required=True, choices=["plus", "minus"])
    parser.add_argument("--accession", default=None, help="Process a single accession only")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_distance_bp", type=int, default=150)
    parser.add_argument("--prominence", type=float, default=0.1)
    parser.add_argument("--max_upstream_dist", type=int, default=100,
                        help="Maximum distance from TSS to downstream gene (bp)")
    args = parser.parse_args()

    if args.accession is not None:
        out_path = filter_and_merge_to_file(
            args.genome_dir, args.predictions_dir, args.output_dir,
            args.accession, args.strand,
            threshold=args.threshold, min_distance_bp=args.min_distance_bp,
            prominence=args.prominence, max_upstream_dist=args.max_upstream_dist)
        print(f"Wrote {out_path}")
    else:
        results = filter_and_merge_all(
            args.genome_dir, args.predictions_dir, args.output_dir, args.strand,
            threshold=args.threshold, min_distance_bp=args.min_distance_bp,
            prominence=args.prominence, max_upstream_dist=args.max_upstream_dist)
        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_err = sum(1 for r in results if r["status"] == "error")
        print(f"{len(results)} files: {n_ok} ok, {n_err} failed")
        for r in results:
            fname = os.path.basename(r["csv_path"])
            if r["status"] == "ok":
                print(f"  OK    {fname} -> {r['out_path']}")
            else:
                print(f"  ERROR {fname}: {r['error_message']}")

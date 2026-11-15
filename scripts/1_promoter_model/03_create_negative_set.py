import argparse
import csv
import random

from Bio import SeqIO


def load_valid_cds(gff_file, fasta_records, offset=100, window=100):
    genes = []
    with open(gff_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            scaffold, feature_type, strand = parts[0], parts[2], parts[6]
            # skip the first 100 bp of each CDS to stay away from the 5' end
            start = int(parts[3]) + offset
            end = int(parts[4])
            if feature_type == "CDS" and strand == "+" and end - start + 1 >= window:
                if scaffold in fasta_records:
                    genes.append((scaffold, start, end))
    return genes


def parse_random_gene_sequences(fasta_file, gff_file, target_number, output_csv):
    fasta_records = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    valid_genes = load_valid_cds(gff_file, fasta_records)
    print(f"{len(valid_genes)} usable + strand CDS")

    results = []
    used = {scaffold: [] for scaffold in fasta_records}

    while len(results) < target_number:
        added = 0
        random.shuffle(valid_genes)

        for scaffold, g_start, g_end in valid_genes:
            if len(results) >= target_number:
                break

            for _ in range(50):
                w_start = random.randint(g_start, g_end - 99)
                w_end = w_start + 99

                if any(max(w_start, u_s) <= min(w_end, u_e) for u_s, u_e in used[scaffold]):
                    continue

                seq = str(fasta_records[scaffold].seq[w_start - 1:w_end]).upper()
                if "N" in seq:
                    continue

                results.append((scaffold, w_start, seq))
                used[scaffold].append((w_start, w_end))
                added += 1
                break

        if added == 0:
            print(f"No more non-overlapping windows, stopped at {len(results)}")
            break

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scaffold", "Start", "Sequence"])
        writer.writerows(results)
    print(f"Wrote {len(results)} sequences to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample 100 bp CDS windows as the negative set")
    parser.add_argument("--fasta", default="reference_Bacil/licheniformis/GCF_034478925.1_ASM3447892v1_genomic.fna")
    parser.add_argument("--gff", default="reference_Bacil/licheniformis/genomic.gff")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--output", default="licheniformis_negative_sequences_revised.csv")
    args = parser.parse_args()
    parse_random_gene_sequences(args.fasta, args.gff, args.n, args.output)

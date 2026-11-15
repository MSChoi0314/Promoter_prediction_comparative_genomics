import argparse
import glob
import os

import pandas as pd

WINDOW = 100
STEP = 10


def read_scaffolds(fasta_path):
    scaffolds = []
    header, seq_lines = None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    scaffolds.append((header, "".join(seq_lines).upper()))
                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        scaffolds.append((header, "".join(seq_lines).upper()))
    return scaffolds


def find_fasta(accession_dir):
    files = []
    for ext in ("*.fasta", "*.fa", "*.fna", "*.fas"):
        files.extend(glob.glob(os.path.join(accession_dir, ext)))
    return files[0] if files else None


def process_fasta_directory(root_directory):
    accession_dirs = glob.glob(os.path.join(root_directory, "GCF_*"))
    if not accession_dirs:
        print(f"No GCF_* folders in {root_directory}")
        return

    for accession_dir in accession_dirs:
        if not os.path.isdir(accession_dir):
            continue
        accession = os.path.basename(accession_dir)
        fasta_file = find_fasta(accession_dir)
        if fasta_file is None:
            print(f"{accession}: no FASTA file")
            continue

        try:
            rows = []
            global_idx = 1
            for scaffold, seq in read_scaffolds(fasta_file):
                for start in range(0, len(seq) - WINDOW + 1, STEP):
                    rows.append({
                        "window_start_global": global_idx,
                        "scaffold": scaffold,
                        "start_in_scaffold": start + 1,
                        "sequence": seq[start:start + WINDOW],
                    })
                    global_idx += 1

            out_csv = os.path.join(accession_dir, f"{accession}_sequences.csv")
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{accession}: {len(rows)} windows")
        except Exception as e:
            print(f"{accession}: error {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split genomes into 100 bp windows (10 bp step), forward strand")
    parser.add_argument("root", nargs="?", default="ncbi_dataset_non-polar/data",
                        help="Folder containing GCF_* subfolders")
    args = parser.parse_args()
    if not os.path.exists(args.root):
        raise SystemExit(f"{args.root} does not exist")
    process_fasta_directory(args.root)

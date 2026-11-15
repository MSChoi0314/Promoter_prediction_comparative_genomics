import argparse
import csv
import os
import sys


def protein_seq_columns(fieldnames):
    cols = [c for c in fieldnames if c.startswith("protein_seq_")]
    return sorted(cols, key=lambda c: int(c.split("_")[2]))


def parse_names(value):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return [n.strip() for n in value.split(",") if n.strip()]


def read_entries(csv_path, prefix):
    entries = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} is empty")
        reader.fieldnames = [c.replace("\ufeff", "") for c in reader.fieldnames]
        seq_cols = protein_seq_columns(reader.fieldnames)
        if not seq_cols:
            raise ValueError(f"{csv_path} has no protein_seq_ columns")

        for row_num, row in enumerate(reader, start=2):
            names = parse_names(row.get("operon_protein_names", ""))
            if not names:
                continue
            seqs = [row.get(c, "").strip() for c in seq_cols]
            n_seq = sum(1 for s in seqs if s)
            if len(names) != n_seq:
                raise ValueError(f"{csv_path} row {row_num}: {len(names)} protein names "
                                 f"but {n_seq} sequences ({names})")
            for name, seq in zip(names, seqs):
                if seq:
                    entries.append((f">{prefix}_{name}", seq))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Collect operon proteins from 01_promoter_to_operon outputs into one FASTA")
    parser.add_argument("input_dir")
    parser.add_argument("output_fasta")
    parser.add_argument("--prefix", default="polar", help="Header prefix, e.g. polar, nonpolar, 9F")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"{args.input_dir} is not a folder")
    csv_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".csv"))
    if not csv_files:
        sys.exit(f"No CSV files in {args.input_dir}")

    all_entries = []
    for name in csv_files:
        try:
            entries = read_entries(os.path.join(args.input_dir, name), args.prefix)
        except ValueError as e:
            sys.exit(f"Stopped: {e}")
        print(f"{name}: {len(entries)} proteins")
        all_entries.extend(entries)

    with open(args.output_fasta, "w", encoding="utf-8") as out:
        for header, seq in all_entries:
            out.write(f"{header}\n{seq}\n")
    print(f"Wrote {len(all_entries)} proteins to {args.output_fasta}")


if __name__ == "__main__":
    main()

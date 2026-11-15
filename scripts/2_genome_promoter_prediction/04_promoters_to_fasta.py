import argparse
import csv
import glob
import os

REQUIRED = {"promoter_probability", "sequence", "scaffold", "start_in_scaffold"}


def process_csv_to_fasta(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*_sequences_*_filtered.csv")))
    if not csv_files:
        print(f"No *_sequences_*_filtered.csv files in {input_dir}")
        return

    for path in csv_files:
        filename = os.path.basename(path)
        accession = filename.split("_sequences")[0]
        is_reverse = "_sequences_reverse" in filename
        tag = "_reverse" if is_reverse else ""
        fasta_path = os.path.join(output_dir, f"{accession}{tag}.fasta")

        with open(path, encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                print(f"{filename}: missing columns {missing}, skipped")
                continue

            count = 0
            with open(fasta_path, "w", encoding="utf-8") as fout:
                for row in reader:
                    if row["promoter_probability"].strip() != "1":
                        continue
                    header = f">{accession}_{row['scaffold']}_{row['start_in_scaffold']}{tag}"
                    fout.write(f"{header}\n{row['sequence']}\n")
                    count += 1

        print(f"{filename}: {count} promoters -> {os.path.basename(fasta_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write promoter calls (promoter_probability == 1) to FASTA")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    process_csv_to_fasta(args.input_dir, args.output_dir)

import argparse

import pandas as pd


def read_table(path):
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def load_cds(gff_path):
    rows = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.strip().split("\t")
            if len(cols) >= 8 and cols[2] == "CDS":
                rows.append({"scaffold": cols[0], "start": int(cols[3]),
                             "end": int(cols[4]), "strand": cols[6]})
    return pd.DataFrame(rows)


def filter_promoters_and_calc_distances(gff_path, promoter_path, output_path):
    cds_df = load_cds(gff_path)
    promoters_df = read_table(promoter_path)

    scaf_col = promoters_df.columns[0]
    strand_col = promoters_df.columns[2]

    kept = []
    for _, row in promoters_df.iterrows():
        scaffold = row[scaf_col]
        pos = int(row["pos_1based"])
        strand = row[strand_col]

        same = cds_df[(cds_df["scaffold"] == scaffold) & (cds_df["strand"] == strand)]
        inside_cds = (same["start"] <= pos) & (same["end"] >= pos)
        if inside_cds.any():
            continue

        before = same[same["end"] < pos]
        after = same[same["start"] > pos]
        dist_up = pd.NA
        dist_down = pd.NA

        if strand == "+":
            if not before.empty:
                dist_up = pos - before["end"].max()
            if not after.empty:
                dist_down = after["start"].min() - pos
        elif strand == "-":
            if not after.empty:
                dist_up = after["start"].min() - pos
            if not before.empty:
                dist_down = pos - before["end"].max()

        rec = row.to_dict()
        rec["dist_to_upstream_CDS"] = dist_up
        rec["dist_to_downstream_CDS"] = dist_down
        kept.append(rec)

    out = pd.DataFrame(kept)
    out.to_excel(output_path, index=False)
    print(f"Input promoters: {len(promoters_df)}")
    print(f"Kept (outside CDS): {len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove TSSs inside CDS and add distances to flanking CDS")
    parser.add_argument("--gff", default="reference_Bacil/licheniformis/genomic.gff")
    parser.add_argument("--promoters", default="licheniformis_promoters_revised.xlsx",
                        help="TSS_promoters.tsv from 01_detect_tss.py (.xlsx, .tsv or .csv)")
    parser.add_argument("--output", default="filtered_promoters_with_distances.xlsx")
    args = parser.parse_args()
    filter_promoters_and_calc_distances(args.gff, args.promoters, args.output)

import argparse

import pandas as pd


def normalize_bit_scores(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df[df["Query Header"].astype(str).str.endswith("_bit-score")].copy()
    value_cols = df.columns[3:]
    df[value_cols] = df[value_cols].div(df["Self-Bit-Score"], axis=0)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Divide BLASTP bit-scores by the self bit-score")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    normalize_bit_scores(args.input_csv, args.output_csv)

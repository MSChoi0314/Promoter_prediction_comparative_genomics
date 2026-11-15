import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
ACGT_SET = set(BASE_TO_IDX.keys())

SEQ_COL_FALLBACKS = ["sequence", "promoter_seq", "seq"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict promoter probability for sequence windows"
    )
    p.add_argument("--input_dir", required=True, help="Directory containing the sequence CSV files")
    p.add_argument("--model_path", required=True, help="Path to the trained .keras model file")
    p.add_argument("--output_dir", required=True, help="Directory to write the annotated CSVs into")
    p.add_argument("--pattern", default="*.csv", help="Input file pattern")
    p.add_argument("--seq_col", default="sequence", help="Sequence column")
    p.add_argument("--prob_col", default="promoter_probability",
                    help="Output probability column")
    p.add_argument("--batch_size", type=int, default=4096, help="Sequences per model.predict() call")
    p.add_argument("--chunksize", type=int, default=None,
                    help="Read CSVs in chunks of this many rows")
    p.add_argument("--drop-invalid", action="store_true", dest="drop_invalid",
                    help="Drop rows with non-ACGT or wrong-length sequences")
    p.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output files")
    p.add_argument("--cpu_only", action="store_true",
                    help="Disable GPU")
    p.add_argument("--num_cpus", type=int, default=None,
                    help="TensorFlow CPU threads")
    p.add_argument("--suffix", default="_result",
                    help="Output filename suffix")
    return p.parse_args()


def build_lookup_table():
    table = np.full(256, 255, dtype=np.uint8)
    for ch, idx in BASE_TO_IDX.items():
        table[ord(ch)] = idx
    return table


LOOKUP_TABLE = build_lookup_table()
EYE4 = np.eye(4, dtype=np.float32)


def one_hot_encode(sequences, seq_len):
    byte_arr = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    byte_arr = byte_arr.reshape(len(sequences), seq_len)
    idx_arr = LOOKUP_TABLE[byte_arr]
    return EYE4[idx_arr]


def resolve_seq_col(df, requested_col, file_label):
    if requested_col in df.columns:
        return requested_col
    for fallback in SEQ_COL_FALLBACKS:
        if fallback in df.columns:
            print(f"  {file_label}: using column '{fallback}'")
            return fallback
    print(f"  {file_label}: no sequence column found, skipped. "
          f"Looked for '{requested_col}' and {SEQ_COL_FALLBACKS}. "
          f"Available columns: {list(df.columns)}")
    return None


def predict_probabilities(model, seqs, seq_len, batch_size):
    out = np.empty(len(seqs), dtype=np.float32)
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        X = one_hot_encode(batch, seq_len)
        out[start:start + len(batch)] = model.predict(X, verbose=0).ravel()
    return out


def annotate_dataframe(df, seq_col, prob_col, model, seq_len, batch_size, drop_invalid):
    seqs_upper = df[seq_col].astype(str).str.upper()

    len_ok = seqs_upper.str.len() == seq_len
    chars_ok = seqs_upper.apply(lambda s: set(s) <= ACGT_SET)
    valid_mask = (len_ok & chars_ok).to_numpy()

    n_total = len(df)
    n_bad_chars = int((~chars_ok).sum())
    n_bad_len = int((~len_ok & chars_ok).sum())
    n_valid = int(valid_mask.sum())

    probs = np.full(n_total, np.nan, dtype=np.float32)
    if n_valid > 0:
        valid_seqs = seqs_upper.to_numpy()[valid_mask].tolist()
        probs[valid_mask] = predict_probabilities(model, valid_seqs, seq_len, batch_size)

    df = df.copy()
    df[prob_col] = probs

    if drop_invalid:
        df = df[valid_mask].reset_index(drop=True)

    stats = {
        "total": n_total,
        "valid": n_valid,
        "skipped_bad_chars": n_bad_chars,
        "skipped_wrong_length": n_bad_len,
    }
    return df, stats


def process_file(path, out_path, model, seq_len, args):
    file_label = os.path.basename(path)
    print(f"Processing {file_label} ...")

    if args.chunksize is None:
        df = pd.read_csv(path, encoding="utf-8-sig")
        seq_col = resolve_seq_col(df, args.seq_col, file_label)
        if seq_col is None:
            return None
        df_out, stats = annotate_dataframe(
            df, seq_col, args.prob_col, model, seq_len, args.batch_size, args.drop_invalid
        )
        df_out.to_csv(out_path, index=False)
    else:
        reader = pd.read_csv(path, encoding="utf-8-sig", chunksize=args.chunksize)
        stats = {"total": 0, "valid": 0, "skipped_bad_chars": 0, "skipped_wrong_length": 0}
        first_chunk = True
        seq_col = None
        for chunk in reader:
            if seq_col is None:
                seq_col = resolve_seq_col(chunk, args.seq_col, file_label)
                if seq_col is None:
                    return None
            chunk_out, chunk_stats = annotate_dataframe(
                chunk, seq_col, args.prob_col, model, seq_len, args.batch_size, args.drop_invalid
            )
            chunk_out.to_csv(out_path, index=False, mode="w" if first_chunk else "a", header=first_chunk)
            first_chunk = False
            for k in stats:
                stats[k] += chunk_stats[k]

    print(f"  total={stats['total']}  valid/predicted={stats['valid']}  "
          f"skipped(non-ACGT)={stats['skipped_bad_chars']}  "
          f"skipped(wrong length)={stats['skipped_wrong_length']}")
    print(f"  -> {out_path}\n")
    return stats


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"--input_dir does not exist or is not a directory: {args.input_dir}")
    if not os.path.isfile(args.model_path):
        sys.exit(f"--model_path does not exist: {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        sys.exit(f"No files matching '{args.pattern}' found in {args.input_dir}")
    print(f"{len(files)} files")

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if args.num_cpus is not None:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.num_cpus))
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(args.num_cpus))
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(args.num_cpus))

    import tensorflow as tf
    from tensorflow import keras
    tf.get_logger().setLevel("ERROR")

    if args.num_cpus is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.num_cpus)
        tf.config.threading.set_inter_op_parallelism_threads(args.num_cpus)

    if args.cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass

    print(f"Model: {args.model_path}")
    model = keras.models.load_model(args.model_path)
    seq_len = model.input_shape[1]
    print(f"Input length: {seq_len}")

    totals = {"total": 0, "valid": 0, "skipped_bad_chars": 0, "skipped_wrong_length": 0}
    n_files_done = 0

    for path in files:
        base = os.path.basename(path)
        stem, ext = os.path.splitext(base)
        out_name = f"{stem}{args.suffix}{ext}"
        out_path = os.path.join(args.output_dir, out_name)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{out_path} exists, skipped")
            continue
        stats = process_file(path, out_path, model, seq_len, args)
        if stats is None:
            continue
        n_files_done += 1
        for k in totals:
            totals[k] += stats[k]

        print(f"Processed {n_files_done}/{len(files)} files")
    print(f"Total rows: {totals['total']}")
    print(f"Predicted: {totals['valid']}")
    print(f"Skipped (non-ACGT): {totals['skipped_bad_chars']}")
    print(f"Skipped (length): {totals['skipped_wrong_length']}")


if __name__ == "__main__":
    main()

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

parser = argparse.ArgumentParser(description="Evaluate saved .keras models on the test set")
parser.add_argument("--test_csv", type=str, required=True, help="Test CSV (promoter_seq, is_promoter)")
parser.add_argument("--model_path", type=str, required=True, help="A .keras file or a folder of .keras files")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--n_threads", type=int, default=None)
args = parser.parse_args()

import tensorflow as tf
from tensorflow import keras

if args.n_threads is not None:
    tf.config.threading.set_intra_op_parallelism_threads(args.n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.n_threads)

os.makedirs(args.output_dir, exist_ok=True)

df = pd.read_csv(args.test_csv)
seqs = df["promoter_seq"].str.upper().tolist()
y_true = df["is_promoter"].values
seq_len = len(seqs[0])
assert all(len(s) == seq_len for s in seqs), "test sequences differ in length"
assert set("".join(seqs)) <= set("ACGT"), "non-ACGT characters in test sequences"
print(f"{len(seqs)} test sequences ({int(y_true.sum())} promoter, {int((1 - y_true).sum())} non-promoter)")

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(sequences):
    arr = np.zeros((len(sequences), len(sequences[0]), 4), dtype=np.float32)
    for i, s in enumerate(sequences):
        for j, ch in enumerate(s):
            arr[i, j, BASE_TO_IDX[ch]] = 1.0
    return arr


X_test = one_hot_encode(seqs)

if os.path.isdir(args.model_path):
    model_paths = sorted(glob.glob(os.path.join(args.model_path, "*.keras")))
    if not model_paths:
        raise SystemExit(f"No .keras files in {args.model_path}")
elif os.path.isfile(args.model_path) and args.model_path.endswith(".keras"):
    model_paths = [args.model_path]
else:
    raise SystemExit(f"Invalid --model_path: {args.model_path}")


def evaluate_one(model_path):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model = keras.models.load_model(model_path)

    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        "model_name": model_name,
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }

    out_path = os.path.join(args.output_dir, f"{model_name}_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{model_name}: TP={tp} FP={fp} FN={fn} TN={tn} "
          f"AUC={result['AUC']:.4f} acc={result['accuracy']:.4f} "
          f"precision={result['precision']:.4f} recall={result['recall']:.4f}")
    return result


all_results = [evaluate_one(p) for p in model_paths]
all_results.sort(key=lambda r: r["accuracy"], reverse=True)

print("\nBy accuracy:")
for r in all_results:
    print(f"  {r['model_name']:40s} acc={r['accuracy']:.4f}  AUC={r['AUC']:.4f}")

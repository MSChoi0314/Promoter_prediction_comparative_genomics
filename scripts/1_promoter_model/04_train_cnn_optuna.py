import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              matthews_corrcoef, recall_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser(description="Optuna search for the promoter CNN")
parser.add_argument("--csv_path", type=str, required=True,
                     help="Training CSV (promoter_seq, is_promoter)")
parser.add_argument("--output_dir", type=str, required=True,
                     help="Output folder for saved trials")
parser.add_argument("--n_threads", type=int, default=None,
                     help="TensorFlow CPU threads")
parser.add_argument("--n_trials", type=int, default=150,
                     help="Number of Optuna trials")
parser.add_argument("--save_threshold", type=float, default=0.90,
                     help="Minimum validation accuracy for saving a trial")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

if args.n_threads is not None:
    tf.config.threading.set_intra_op_parallelism_threads(args.n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.n_threads)

RANDOM_SEED = args.seed
CSV_PATH = args.csv_path
OUTPUT_DIR = args.output_dir
N_TRIALS = args.n_trials
SAVE_THRESHOLD = args.save_threshold
USE_KFOLD_OBJECTIVE = False
N_FOLDS = 5

rng_np = np.random.default_rng(RANDOM_SEED)

df = pd.read_csv(CSV_PATH)
seqs = df["promoter_seq"].str.upper().tolist()
y = df["is_promoter"].values
seq_len = len(seqs[0])


def build_clusters(sequences, k=20, stride=5):
    kmer_to_idx = defaultdict(set)
    for i, s in enumerate(sequences):
        for j in range(0, len(s) - k + 1, stride):
            kmer_to_idx[s[j:j + k]].add(i)
    parent = list(range(len(sequences)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for idxs in kmer_to_idx.values():
        idxs = list(idxs)
        for i in range(1, len(idxs)):
            union(idxs[0], idxs[i])
    clusters = defaultdict(list)
    for i in range(len(sequences)):
        clusters[find(i)].append(i)
    return list(clusters.values())


def cluster_aware_split(sequences, fracs, seed=RANDOM_SEED):
    clusters = build_clusters(sequences)
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)
    n = len(sequences)
    targets = {name: int(n * f) for name, f in fracs.items()}
    buckets = {name: [] for name in fracs}
    buckets["train"] = []
    for c in clusters:
        placed = False
        for name in fracs:
            if len(buckets[name]) < targets[name]:
                buckets[name].extend(c)
                placed = True
                break
        if not placed:
            buckets["train"].extend(c)
    return {name: np.array(idx) for name, idx in buckets.items()}


splits = cluster_aware_split(seqs, {"test": 0.15, "val": 0.15})
train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]
print(f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(sequences):
    arr = np.zeros((len(sequences), len(sequences[0]), 4), dtype=np.float32)
    for i, s in enumerate(sequences):
        for j, ch in enumerate(s):
            arr[i, j, BASE_TO_IDX[ch]] = 1.0
    return arr


X_all = one_hot_encode(seqs)
X_train, y_train = X_all[train_idx], y[train_idx]
X_val, y_val = X_all[val_idx], y[val_idx]
X_test, y_test = X_all[test_idx], y[test_idx]

def build_model(seq_len, n_filters, kernel_size, dropout, l2_reg,
                 dense_units, learning_rate):
    reg = keras.regularizers.l2(l2_reg)
    model = keras.Sequential([
        keras.Input(shape=(seq_len, 4)),
        layers.Conv1D(n_filters, kernel_size, activation="relu",
                       padding="same", kernel_regularizer=reg),
        layers.Dropout(dropout),
        layers.GlobalMaxPooling1D(),
        layers.Dense(dense_units, activation="relu", kernel_regularizer=reg),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


class PruningCallback(keras.callbacks.Callback):
    def __init__(self, trial, monitor="val_accuracy"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        score = logs.get(self.monitor)
        if score is None:
            return
        self.trial.report(float(score), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}")


def save_trial_artifacts(trial_number, hp, score, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"trial{trial_number}_{timestamp}"

    model_path = os.path.join(output_dir, f"{base}.keras")
    model.save(model_path)

    params_path = os.path.join(output_dir, f"{base}.para.json")
    payload = {
        "trial_number": trial_number,
        "validation_accuracy": float(score),
        "hyperparameters": hp,
        "saved_at": timestamp,
    }
    with open(params_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  saved {base} ({score:.4f})")


def retrain_for_saving(hp):
    from sklearn.model_selection import train_test_split
    pool_idx = np.concatenate([train_idx, val_idx])
    idx_tr, idx_va = train_test_split(
        pool_idx, test_size=0.15, stratify=y[pool_idx], random_state=RANDOM_SEED)
    _, model = train_and_eval(hp, X_all[idx_tr], y[idx_tr],
                               X_all[idx_va], y[idx_va], trial=None)
    return model


def train_and_eval(hp, Xtr, ytr, Xva, yva, trial=None, epochs=80):
    model_kwargs = {k: v for k, v in hp.items() if k != "batch_size"}
    model = build_model(seq_len=seq_len, **model_kwargs)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True),
    ]
    if trial is not None:
        callbacks.append(PruningCallback(trial, monitor="val_accuracy"))
    model.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=epochs, batch_size=hp["batch_size"],
              callbacks=callbacks, verbose=0)
    val_acc = model.evaluate(Xva, yva, verbose=0)[1]
    return val_acc, model


def objective(trial):
    hp = {
        "n_filters": trial.suggest_int("n_filters", 256, 1024, step=256),
        "kernel_size": trial.suggest_int("kernel_size", 15, 30),
        "dropout": trial.suggest_float("dropout", 0.1, 0.7, step=0.1),
        "l2_reg": trial.suggest_float("l2_reg", 1e-7, 1e-4, log=True),
        "dense_units": trial.suggest_int("dense_units", 8, 32, step=8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
    }

    if USE_KFOLD_OBJECTIVE:
        from sklearn.model_selection import StratifiedKFold
        pool_idx = np.concatenate([train_idx, val_idx])
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        accs = []
        for fold_train, fold_val in skf.split(pool_idx, y[pool_idx]):
            acc, _ = train_and_eval(
                hp,
                X_all[pool_idx[fold_train]], y[pool_idx[fold_train]],
                X_all[pool_idx[fold_val]], y[pool_idx[fold_val]],
                trial=None,
            )
            accs.append(acc)
        score = float(np.mean(accs))
        model = None
    else:
        score, model = train_and_eval(hp, X_train, y_train, X_val, y_val, trial=trial)

    if score >= SAVE_THRESHOLD:
        model_to_save = model if model is not None else retrain_for_saving(hp)
        save_trial_artifacts(trial.number, hp, score, model_to_save, OUTPUT_DIR)

    return score


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15),
)
study.optimize(objective, n_trials=N_TRIALS)

print("\nBest trial:")
print(f"  validation accuracy = {study.best_value:.4f}")
print(f"  params = {study.best_params}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "best_hyperparams.json"), "w") as f:
    json.dump(study.best_params, f, indent=2)

X_trainval = np.concatenate([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

final_model_kwargs = {k: v for k, v in study.best_params.items() if k != "batch_size"}
final_model = build_model(seq_len=seq_len, **final_model_kwargs)
final_model.fit(
    X_trainval, y_trainval,
    validation_split=0.1,
    epochs=150,
    batch_size=study.best_params["batch_size"],
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                              restore_best_weights=True)],
    verbose=2,
)

y_prob = final_model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"\nTest set (n={len(y_test)})")
print(f"  Accuracy    = {accuracy_score(y_test, y_pred):.3f}")
print(f"  Sensitivity = {recall_score(y_test, y_pred):.3f}")
print(f"  Specificity = {tn / (tn + fp):.3f}")
print(f"  MCC         = {matthews_corrcoef(y_test, y_pred):.3f}")
print(f"  Total trainable params = {final_model.count_params():,}")

final_model.save(os.path.join(OUTPUT_DIR, "promoter_cnn_final_best.keras"))

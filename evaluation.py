"""Evaluation: every model goes through the same splits and gets the same scores.

Splits (the same as the original research):
  * 2-fold cross-validation over the pairs (KFold without shuffling, which is what
    cross_val_score(cv=2) did in the original code);
  * train/test: train_test_split(test_size=0.3, random_state=50).

Scores:
  * R², MAE and MSE on the 15-digit encoding (the original research's metrics);
  * nucleotide accuracy: how many nucleotides the model got right;
  * mutation accuracy: accuracy only on the positions that changed between the old and
    the new sequence, which is the hard part to predict.
"""
import time

import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split

from dataset import IGNORE_INDEX, pad_matrix
from encoding_data import CHUNK_SIZE, EncodingData

_encoder = EncodingData()


def decode_chunks(row, true_digits):
    """Turns predicted 15-digit numbers back into digits (nucleotides).

    This lets the classical models be scored nucleotide by nucleotide too. A chunk that does
    not turn into a valid number of the right length counts as entirely wrong.
    """
    out = []
    for j in range(0, len(true_digits), CHUNK_SIZE):
        size = min(CHUNK_SIZE, len(true_digits) - j)
        value = row[j // CHUNK_SIZE]
        text = str(int(round(value))).zfill(size) if np.isfinite(value) and value >= 0 else ''
        out.append(text if len(text) == size else '0' * size)
    return ''.join(out)


class ChunkModel:
    """Wraps a scikit-learn model that works on the 15-digit encoding."""

    def __init__(self, name, factory):
        self.name, self.factory = name, factory

    def fit_predict(self, data, train_idx, test_idx):
        model = self.factory()
        model.fit(data.X[train_idx], data.Y[train_idx])
        chunks = np.asarray(model.predict(data.X[test_idx]), dtype=np.float64).reshape(len(test_idx), -1)
        digits = [decode_chunks(chunks[r], data.target_digits[i]) for r, i in enumerate(test_idx)]
        return digits, chunks


class SequenceModel:
    """Wraps a model that predicts one nucleotide per position (class 0-3 = A, T, G, C)."""

    def __init__(self, name, factory):
        self.name, self.factory = name, factory

    def fit_predict(self, data, train_idx, test_idx):
        model = self.factory()
        model.fit(data.X_tokens[train_idx], data.Y_classes[train_idx])
        classes = np.asarray(model.predict(data.X_tokens[test_idx]))
        digits = []
        for r, i in enumerate(test_idx):
            scored = data.Y_classes[i] != IGNORE_INDEX
            digits.append(''.join(str(c + 1) for c in classes[r][scored]))
        # Also convert to the 15-digit encoding, so R², MAE and MSE can be computed
        chunks = pad_matrix([_encoder.chunk(d) for d in digits], data.chunk_width)
        return digits, chunks


class CopyBaseline:
    """Baseline model: says next season's sequence is the same as this season's."""

    def fit(self, X, Y):
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.where((X >= 1) & (X <= 4), X - 1, 0)


def score(data, test_idx, pred_digits, pred_chunks):
    """Computes every score for one test set."""
    Y_true = data.Y[test_idx]
    matches = total = mut_hits = mut_total = 0
    for r, i in enumerate(test_idx):
        true = np.frombuffer(data.target_digits[i].encode(), dtype=np.uint8)
        pred = np.frombuffer(pred_digits[r].encode(), dtype=np.uint8)
        equal = true == pred
        sites = data.mutation_sites[i]
        matches += int(equal.sum())
        total += len(true)
        mut_hits += int(equal[sites].sum())
        mut_total += len(sites)
    return {
        'r2': metrics.r2_score(Y_true, pred_chunks),
        'r2_variance_weighted': metrics.r2_score(Y_true, pred_chunks, multioutput='variance_weighted'),
        'mae': metrics.mean_absolute_error(Y_true, pred_chunks),
        'mse': metrics.mean_squared_error(Y_true, pred_chunks),
        'nt_accuracy': matches / total,
        'mutation_accuracy': mut_hits / mut_total if mut_total else float('nan'),
        'mutation_sites': mut_total,
    }


def evaluate(model, data, cv_folds=2, test_size=0.3, random_state=50):
    """Runs a model through cross-validation and the train/test split, and returns every score."""
    indices = np.arange(len(data))

    cv_r2, cv_acc = [], []
    for train_idx, test_idx in KFold(n_splits=cv_folds).split(indices):
        digits, chunks = model.fit_predict(data, train_idx, test_idx)
        fold = score(data, test_idx, digits, chunks)
        cv_r2.append(fold['r2'])
        cv_acc.append(fold['nt_accuracy'])

    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    start = time.perf_counter()
    digits, chunks = model.fit_predict(data, train_idx, test_idx)
    seconds = time.perf_counter() - start

    return {
        'dataset': data.name,
        'model': model.name,
        'cv_r2': cv_r2,
        'cv_nt_accuracy': cv_acc,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'seconds': seconds,
        **score(data, test_idx, digits, chunks),
    }

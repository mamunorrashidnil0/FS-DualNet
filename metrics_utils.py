"""
Threshold utilities (numpy only).

Why: under a generator shift the model's ranking (AUC/AP) usually stays good while the fixed 0.5 threshold stops fitting,
so ACC@0.5 falls far below what AUC suggests. Three honest ways to report it:
  * ACC@0.5                 : fixed threshold (what DIRE reports)
  * ACC@dev-threshold       : threshold picked on a DEV set from a DIFFERENT generator than the training one (no test labels used) -> legitimate
  * ACC@oracle              : best threshold on the test set itself -> UPPER BOUND ONLY, uses test labels, never report it as the accuracy
"""
import numpy as np


def best_threshold(y, p, balanced=False):
    """Return (thr, score): predict fake when p >= thr; maximises accuracy (or balanced accuracy)."""
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=float)
    n = len(y)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    neg = np.concatenate([[0], np.cumsum(1 - ys)])  # reals among the first k samples (they get predicted real)
    pos = np.concatenate([[0], np.cumsum(ys)])      # fakes among the first k samples (wrongly predicted real)
    n_pos, n_neg = int(pos[-1]), int(neg[-1])
    tn, tp = neg, n_pos - pos
    score = 0.5 * (tn / max(n_neg, 1) + tp / max(n_pos, 1)) if balanced else (tn + tp) / n
    valid = np.ones(n + 1, dtype=bool)
    valid[1:n] = ps[:-1] < ps[1:]  # cut only between different score values
    score = np.where(valid, score, -1.0)
    k = int(score.argmax())
    thr = 0.0 if k == 0 else (1.0 + 1e-6 if k == n else float((ps[k - 1] + ps[k]) / 2))
    return thr, float(score[k])


def acc_at(y, p, thr):
    return float(((np.asarray(p) >= thr).astype(int) == np.asarray(y).astype(int)).mean())


def bacc_at(y, p, thr):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else 0.0
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else 0.0
    return float(0.5 * (tpr + tnr))

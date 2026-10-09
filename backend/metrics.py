import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.0
    return float(average_precision_score(y, s))


def recall_at_fpr(y, s, fpr=0.01):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(s); ys = y[order]
    nneg = max(1, (y == 0).sum()); npos = max(1, (y == 1).sum())
    fprs = np.cumsum(1 - ys) / nneg
    rec = np.cumsum(ys) / npos
    m = fprs <= fpr
    return float(rec[m].max()) if m.any() else 0.0


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.argmax(f1))
    return float(t[min(i, len(t) - 1)]), float(f1[i])


def full_metrics(y, s, thr=0.5):
    y = np.asarray(y); s = np.asarray(s)
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return dict(pr_auc=pr_auc(y, s), recall_at_1pct_fpr=recall_at_fpr(y, s),
                precision=float(prec), recall=float(rec), f1=float(f1),
                confusion=[[tn, fp], [fn, tp]])

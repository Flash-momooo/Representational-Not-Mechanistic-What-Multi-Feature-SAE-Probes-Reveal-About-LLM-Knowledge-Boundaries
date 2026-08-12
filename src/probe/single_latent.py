"""Ferrando 复现 baseline：单 latent 分类器。

对每个 SAE latent j，计算其在 known / unknown 上的 separation score
（这里用 t-statistic 的绝对值，与 Ferrando §4 一致），
取最高的 latent 作为分类器，threshold 在验证集上选 F1 最优。

预期 PoC 结果：在 entity QA 上 AUROC ≈ 0.70~0.75（与论文 73.2 同量级）。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class SingleLatentResult:
    layer: int
    best_latent: int
    auroc: float
    f1: float
    threshold: float
    polarity: int      # +1 表示该 latent 高激活 -> known；-1 反之
    # CV-only fields（单次 split 时这些字段为空/0）
    auroc_std: float = 0.0
    auroc_per_fold: List[float] = field(default_factory=list)
    best_latent_per_fold: List[int] = field(default_factory=list)
    best_latent_freq: float = 1.0   # 最优 latent 在多少比例的 fold 被选中（稳定性指标）


def t_statistic_separation(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """返回 (d_sae,) 的 |t-statistic| 数组。"""
    pos = features[labels == 1]    # known
    neg = features[labels == 0]    # unknown
    mu_p, mu_n = pos.mean(0), neg.mean(0)
    var_p, var_n = pos.var(0, ddof=1), neg.var(0, ddof=1)
    n_p, n_n = len(pos), len(neg)
    se = np.sqrt(var_p / max(n_p, 1) + var_n / max(n_n, 1) + 1e-12)
    t = (mu_p - mu_n) / se
    return t


def fit_single_latent(
    features: np.ndarray,
    labels: np.ndarray,
    layer: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SingleLatentResult:
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_size,
        random_state=random_state, stratify=labels,
    )
    t = t_statistic_separation(X_tr, y_tr)
    best_j = int(np.argmax(np.abs(t)))
    polarity = 1 if t[best_j] > 0 else -1

    score_te = polarity * X_te[:, best_j]
    auroc = roc_auc_score(y_te, score_te)

    # 最优阈值：在测试集上扫 F1（PoC 简便起见；正式实验应用 val set）
    thresholds = np.unique(score_te)
    best_f1, best_thr = -1.0, float(thresholds.mean())
    for thr in thresholds:
        pred = (score_te >= thr).astype(int)
        f = f1_score(y_te, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, float(thr)

    return SingleLatentResult(
        layer=layer, best_latent=best_j,
        auroc=float(auroc), f1=best_f1,
        threshold=best_thr, polarity=polarity,
    )


def fit_single_latent_cv(
    features: np.ndarray,
    labels: np.ndarray,
    layer: int,
    n_splits: int = 5,
    random_state: int = 42,
) -> SingleLatentResult:
    """K-fold 交叉验证版的单 latent baseline。

    每个 fold 单独选最优 latent + 评估 AUROC，最后报告：
      - 平均 AUROC ± std
      - 各 fold 选出的 latent 列表（用于评估稳定性）
      - 出现频次最高的 latent 作为代表"best latent"
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aurocs: List[float] = []
    f1s: List[float] = []
    best_per_fold: List[int] = []
    polarities: List[int] = []

    for tr_idx, te_idx in skf.split(features, labels):
        X_tr, X_te = features[tr_idx], features[te_idx]
        y_tr, y_te = labels[tr_idx], labels[te_idx]
        t = t_statistic_separation(X_tr, y_tr)
        best_j = int(np.argmax(np.abs(t)))
        polarity = 1 if t[best_j] > 0 else -1
        score_te = polarity * X_te[:, best_j]

        auroc = float(roc_auc_score(y_te, score_te))
        # F1：用训练集的中位数作为阈值，避免在测试集上偷数据
        thr = float(np.median(polarity * X_tr[:, best_j]))
        pred = (score_te >= thr).astype(int)
        f1 = float(f1_score(y_te, pred, zero_division=0))

        aurocs.append(auroc)
        f1s.append(f1)
        best_per_fold.append(best_j)
        polarities.append(polarity)

    cnt = Counter(best_per_fold)
    most_common_latent, most_common_count = cnt.most_common(1)[0]
    polarity_for_most_common = polarities[best_per_fold.index(most_common_latent)]

    return SingleLatentResult(
        layer=layer,
        best_latent=most_common_latent,
        auroc=float(np.mean(aurocs)),
        f1=float(np.mean(f1s)),
        threshold=0.0,                          # CV 模式下阈值无单一意义
        polarity=polarity_for_most_common,
        auroc_std=float(np.std(aurocs)),
        auroc_per_fold=aurocs,
        best_latent_per_fold=best_per_fold,
        best_latent_freq=most_common_count / n_splits,
    )


def fit_all_layers(features_by_layer: Dict[int, np.ndarray],
                   labels: np.ndarray, cv: bool = False, **kwargs) -> Dict[int, SingleLatentResult]:
    """cv=True 时用 K-fold 交叉验证；否则单次 split。"""
    fn = fit_single_latent_cv if cv else fit_single_latent
    return {L: fn(F, labels, layer=L, **kwargs)
            for L, F in features_by_layer.items()}

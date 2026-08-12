"""Baseline probes：在原始 residual stream（非 SAE）上跑 LogReg。

用于回答 reviewer "SAE 真的有用吗？" 的质疑：
  - 如果 SAE 与 raw residual 性能相当 → SAE 的核心价值是**可解释性**（top features + Neuronpedia）
  - 如果 SAE 显著更好 → 稀疏化捕捉了关键信号
  - 如果 raw 显著更好 → SAE 编码丢了信息，需要换 SAE 或方法

提供两种 baseline：
  L2-LogReg：标准 dense 线性分类器（最强 baseline）
  L1-LogReg：稀疏化基线（与多特征 SAE probe 同正则）
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineResult:
    layer: int
    method: str          # "raw_l2" | "raw_l1" | ...
    auroc: float
    auprc: float
    f1: float
    auroc_std: float = 0.0
    auroc_per_fold: List[float] = field(default_factory=list)
    n_features: int = 0


def fit_raw_residual_cv(
    raw: np.ndarray,
    labels: np.ndarray,
    layer: int,
    penalty: str = "l2",
    C: float = 1.0,
    n_splits: int = 5,
    random_state: int = 42,
) -> BaselineResult:
    """在原始 residual stream（dense d_model 维）上跑 K-fold CV LogReg。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aurocs, auprcs, f1s = [], [], []

    for tr_idx, te_idx in skf.split(raw, labels):
        X_tr, X_te = raw[tr_idx], raw[te_idx]
        y_tr, y_te = labels[tr_idx], labels[te_idx]
        # raw residual 是 dense 浮点向量，可以做 zero-mean
        scaler = StandardScaler(with_mean=True)
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            solver = "liblinear" if penalty == "l1" else "lbfgs"
            clf = LogisticRegression(
                penalty=penalty, solver=solver,
                C=C, max_iter=2000, random_state=random_state,
            )
            clf.fit(X_tr_s, y_tr)
        proba = clf.predict_proba(X_te_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aurocs.append(float(roc_auc_score(y_te, proba)))
        auprcs.append(float(average_precision_score(y_te, proba)))
        f1s.append(float(f1_score(y_te, pred, zero_division=0)))

    return BaselineResult(
        layer=layer,
        method=f"raw_{penalty}",
        auroc=float(np.mean(aurocs)),
        auprc=float(np.mean(auprcs)),
        f1=float(np.mean(f1s)),
        auroc_std=float(np.std(aurocs)),
        auroc_per_fold=aurocs,
        n_features=raw.shape[1],
    )

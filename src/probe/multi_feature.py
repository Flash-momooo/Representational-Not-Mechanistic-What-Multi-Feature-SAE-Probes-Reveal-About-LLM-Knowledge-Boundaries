"""创新点 β：多特征 KB-Probe（L1-Logistic Regression）。

相对于 Ferrando 单 latent，使用 L1 正则化的 Logistic Regression
在 16k 维稀疏 SAE 特征上训练，自动选出 top-K 信息特征。

可解释性：训练完后输出非零权重特征列表 + Neuronpedia 链接。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


NEURONPEDIA_URL = "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-{width}/{idx}"


@dataclass
class MultiFeatureResult:
    layer: int
    auroc: float
    auprc: float
    f1: float
    n_nonzero_features: int
    top_features: List[dict] = field(default_factory=list)
    # CV-only fields（单次 split 时为空/0）
    auroc_std: float = 0.0
    auroc_per_fold: List[float] = field(default_factory=list)
    n_nonzero_per_fold: List[int] = field(default_factory=list)


def fit_multi_feature(
    features: np.ndarray,
    labels: np.ndarray,
    layer: int,
    width: str = "16k",
    C: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
    top_k: int = 20,
) -> MultiFeatureResult:
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_size,
        random_state=random_state, stratify=labels,
    )

    scaler = StandardScaler(with_mean=False)  # 保留稀疏性
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # liblinear 支持 L1 正则；新版 sklearn 推荐 saga + l1_ratio，但 liblinear 仍有效
    # 显式抑制 sklearn 1.8+ 的 penalty deprecation warning（liblinear 不支持 l1_ratio）
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        clf = LogisticRegression(
            penalty="l1", solver="liblinear",
            C=C, max_iter=2000, random_state=random_state,
        )
        clf.fit(X_tr_s, y_tr)

    proba = clf.predict_proba(X_te_s)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auroc = float(roc_auc_score(y_te, proba))
    auprc = float(average_precision_score(y_te, proba))
    f1 = float(f1_score(y_te, pred, zero_division=0))

    coefs = clf.coef_.ravel()
    nonzero_idx = np.flatnonzero(coefs)
    order = np.argsort(-np.abs(coefs[nonzero_idx]))[:top_k]
    top_idx = nonzero_idx[order]
    top_features = [
        {
            "feature_idx": int(j),
            "weight": float(coefs[j]),
            "neuronpedia": NEURONPEDIA_URL.format(layer=layer, width=width, idx=int(j)),
        }
        for j in top_idx
    ]

    return MultiFeatureResult(
        layer=layer,
        auroc=auroc, auprc=auprc, f1=f1,
        n_nonzero_features=int(len(nonzero_idx)),
        top_features=top_features,
    )


def fit_multi_feature_cv(
    features: np.ndarray,
    labels: np.ndarray,
    layer: int,
    width: str = "16k",
    C: float = 1.0,
    n_splits: int = 5,
    random_state: int = 42,
    top_k: int = 20,
) -> MultiFeatureResult:
    """K-fold CV 版多特征 probe。聚合各 fold 的 AUROC + 选择稳定的 top features。"""
    import warnings
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aurocs, auprcs, f1s, nzs = [], [], [], []
    coef_sum = np.zeros(features.shape[1], dtype=np.float64)
    coef_count = np.zeros(features.shape[1], dtype=np.int32)

    for tr_idx, te_idx in skf.split(features, labels):
        X_tr, X_te = features[tr_idx], features[te_idx]
        y_tr, y_te = labels[tr_idx], labels[te_idx]
        scaler = StandardScaler(with_mean=False)
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            clf = LogisticRegression(
                penalty="l1", solver="liblinear",
                C=C, max_iter=2000, random_state=random_state,
            )
            clf.fit(X_tr_s, y_tr)
        proba = clf.predict_proba(X_te_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aurocs.append(float(roc_auc_score(y_te, proba)))
        auprcs.append(float(average_precision_score(y_te, proba)))
        f1s.append(float(f1_score(y_te, pred, zero_division=0)))
        coef = clf.coef_.ravel()
        nz = np.flatnonzero(coef)
        nzs.append(int(len(nz)))
        coef_sum[nz] += coef[nz]
        coef_count[nz] += 1

    # 选稳定特征：在 ≥ ceil(n/2) 个 fold 都非零，按平均权重排序
    min_folds = (n_splits + 1) // 2
    stable_mask = coef_count >= min_folds
    stable_idx = np.flatnonzero(stable_mask)
    avg_coef = np.where(coef_count > 0, coef_sum / np.maximum(coef_count, 1), 0.0)
    order = np.argsort(-np.abs(avg_coef[stable_idx]))[:top_k]
    top_idx = stable_idx[order]
    top_features = [
        {
            "feature_idx": int(j),
            "weight": float(avg_coef[j]),
            "folds_selected": int(coef_count[j]),
            "neuronpedia": NEURONPEDIA_URL.format(layer=layer, width=width, idx=int(j)),
        }
        for j in top_idx
    ]
    return MultiFeatureResult(
        layer=layer,
        auroc=float(np.mean(aurocs)),
        auprc=float(np.mean(auprcs)),
        f1=float(np.mean(f1s)),
        n_nonzero_features=int(np.mean(nzs)),
        top_features=top_features,
        auroc_std=float(np.std(aurocs)),
        auroc_per_fold=aurocs,
        n_nonzero_per_fold=nzs,
    )


def fit_all_layers(features_by_layer: Dict[int, np.ndarray],
                   labels: np.ndarray, cv: bool = False, **kwargs) -> Dict[int, MultiFeatureResult]:
    """cv=True 时用 K-fold 交叉验证；否则单次 split。"""
    fn = fit_multi_feature_cv if cv else fit_multi_feature
    return {L: fn(F, labels, layer=L, **kwargs)
            for L, F in features_by_layer.items()}

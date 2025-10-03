from typing import List, Dict, Optional, Tuple
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import json
import os


def _standardized_mean_diff(X_t: np.ndarray, X_c: np.ndarray) -> np.ndarray:
    """
    SMD = (mean_t - mean_c) / pooled_std
    """
    mt, mc = X_t.mean(axis=0), X_c.mean(axis=0)
    vt, vc = X_t.var(axis=0, ddof=1), X_c.var(axis=0, ddof=1)
    pooled = np.sqrt((vt + vc) / 2.0 + 1e-12)
    return (mt - mc) / pooled

def psm_att(df: pd.DataFrame, treatment: str, outcome: str, covariates: List[str],
            random_state: int = 42, caliper: Optional[float] = None,
            out_dir: Optional[str] = None, prefix: Optional[str] = None) -> Dict[str, float]:
    """
    dict : {'att','n_pairs','smd_before_mean','smd_after_mean', 'pairs': list}
    """

    X_df = df[covariates].copy()
    
    X = X_df.to_numpy()
    y = df[outcome].to_numpy()
    T = df[treatment].astype(int).to_numpy()

    # Propensity score estimates
    lr = LogisticRegression(max_iter=3000, class_weight=None, n_jobs=None)
    lr.fit(X, T)

    # ps: If this user has features of X, what is the probability that the user responds "yes"
    # P(T=1|X)
    ps = lr.predict_proba(X)[:, 1]
    ps_std = np.std(ps)

    # caliper = 0.2 * ps_std

    # Divide 'Treatment' from 'Control' -> distance based on PS
    idx_t = np.where(T == 1)[0]    # Response=1
    idx_c = np.where(T == 0)[0]    # Response=0
    ps_t, ps_c = ps[idx_t], ps[idx_c]

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(ps_c.reshape(-1, 1))
    dist, idx = nn.kneighbors(ps_t.reshape(-1, 1), return_distance=True)
    idx = idx.ravel()
    dist = dist.ravel()

    pairs: List[Tuple[int, int]] = []
    for i_t, d, i_c_local in zip(idx_t, dist, idx):
        i_c = idx_c[i_c_local]
        # caliper: if too far
        if caliper is not None and d > caliper:
            continue
        pairs.append((int(i_c), int(i_t)))

    # ATT
    diffs = [y[i_t] - y[i_c] for i_c, i_t in pairs]
    att = float(np.mean(diffs)) if len(diffs) > 0 else float("nan")

    # SMD -> Compare before/after matching
    # if X is a sparse matrix, transform into dense matrix
    def _to_dense(Z):
        return Z.toarray() if hasattr(Z, "toarray") else Z
    X_dense = _to_dense(X)
    
    smd_before = _standardized_mean_diff(X_dense[idx_t], X_dense[idx_c])
    smd_before_mean = float(np.mean(np.abs(smd_before)))
    
    smd_before_detail = {cov: float(smd) for cov, smd in zip(covariates, np.abs(smd_before))}
    
    ps_after = {}
    smd_after_detail = {}
    
    if pairs:
        t_sel = np.array([i_t for _, i_t in pairs])
        c_sel = np.array([i_c for i_c, _ in pairs])
        smd_after = _standardized_mean_diff(X_dense[t_sel], X_dense[c_sel])
        smd_after_mean = float(np.mean(np.abs(smd_after)))

        smd_after_detail = {cov: float(smd) for cov, smd in zip(covariates, np.abs(smd_after))}
        ps_after = {
            "treatment": ps[t_sel].tolist(),
            "control": ps[c_sel].tolist()
        }
    else:
        smd_after_mean = float("nan")

    ps_before = {
        "treatment": ps[idx_t].tolist(),
        "control": ps[idx_c].tolist()
    }

    res = {
        "att": att, "n_pairs": int(len(pairs)),
        "caliper": caliper,
        "smd_before_mean": smd_before_mean,
        "smd_after_mean": smd_after_mean,
        "pairs": pairs[:10],
        "smd_before_detail": smd_before_detail,
        "smd_after_detail": smd_after_detail,
        "ps_before": ps_before,
        "ps_after": ps_after,
    }


    if out_dir is not None and prefix is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{prefix}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    return res

# Use XGBoost
def psm_att_xgboost(df: pd.DataFrame, treatment: str, outcome: str, covariates: List[str],
            random_state: int = 42, caliper: Optional[float] = None,
            out_dir: Optional[str] = None, prefix: Optional[str] = None) -> Dict[str, float]:
    """
    dict : {'att','n_pairs','smd_before_mean','smd_after_mean', 'pairs': list}
    """

    X_df = df[covariates].copy()
    
    X = X_df.to_numpy()
    y = df[outcome].to_numpy()
    T = df[treatment].astype(int).to_numpy()

    # Propensity score estimates
    lr = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, 
                       use_label_encoder=False, eval_metric='logloss', random_state=42)
    lr.fit(X, T)

    # ps: If this user has features of X, what is the probability that the user responds "yes"
    # P(T=1|X)
    ps = lr.predict_proba(X)[:, 1]
    ps_std = np.std(ps)

    # caliper = 0.2 * ps_std

    # Divide 'Treatment' from 'Control' -> distance based on PS
    idx_t = np.where(T == 1)[0]    # Response=1
    idx_c = np.where(T == 0)[0]    # Response=0
    ps_t, ps_c = ps[idx_t], ps[idx_c]

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(ps_c.reshape(-1, 1))
    dist, idx = nn.kneighbors(ps_t.reshape(-1, 1), return_distance=True)
    idx = idx.ravel()
    dist = dist.ravel()

    pairs: List[Tuple[int, int]] = []
    for i_t, d, i_c_local in zip(idx_t, dist, idx):
        i_c = idx_c[i_c_local]
        # caliper: if too far
        if caliper is not None and d > caliper:
            continue
        pairs.append((int(i_c), int(i_t)))

    # ATT
    diffs = [y[i_t] - y[i_c] for i_c, i_t in pairs]
    att = float(np.mean(diffs)) if len(diffs) > 0 else float("nan")

    # SMD -> Compare before/after matching
    # if X is a sparse matrix, transform into dense matrix
    def _to_dense(Z):
        return Z.toarray() if hasattr(Z, "toarray") else Z
    X_dense = _to_dense(X)
    
    smd_before = _standardized_mean_diff(X_dense[idx_t], X_dense[idx_c])
    smd_before_mean = float(np.mean(np.abs(smd_before)))
    
    smd_before_detail = {cov: float(smd) for cov, smd in zip(covariates, np.abs(smd_before))}
    
    ps_after = {}
    smd_after_detail = {}
    
    if pairs:
        t_sel = np.array([i_t for _, i_t in pairs])
        c_sel = np.array([i_c for i_c, _ in pairs])
        smd_after = _standardized_mean_diff(X_dense[t_sel], X_dense[c_sel])
        smd_after_mean = float(np.mean(np.abs(smd_after)))

        smd_after_detail = {cov: float(smd) for cov, smd in zip(covariates, np.abs(smd_after))}
        ps_after = {
            "treatment": ps[t_sel].tolist(),
            "control": ps[c_sel].tolist()
        }
    else:
        smd_after_mean = float("nan")

    ps_before = {
        "treatment": ps[idx_t].tolist(),
        "control": ps[idx_c].tolist()
    }

    res = {
        "att": att, "n_pairs": int(len(pairs)),
        "caliper": caliper,
        "smd_before_mean": smd_before_mean,
        "smd_after_mean": smd_after_mean,
        "pairs": pairs[:10],
        "smd_before_detail": smd_before_detail,
        "smd_after_detail": smd_after_detail,
        "ps_before": ps_before,
        "ps_after": ps_after,
    }


    if out_dir is not None and prefix is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{prefix}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    return res
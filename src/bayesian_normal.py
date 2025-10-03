from typing import Dict, Optional, List
import pandas as pd
from scipy.stats import norm
import math


def posterior_normal_unknown_var(mu0: float, kappa0: float, alpha0: float, beta0: float,
                                 xbar: float, s2: float, n: int) -> Dict[str, float]:
    """
    σ^2 unknown: Normal–Inverse-Gamma
    """

    kappa_n = kappa0 + n    # kappa_n: prior(kappa0) + number of data(n)
    mu_n = (kappa0 * mu0 + n * xbar) / kappa_n    # mu_n: posterior average.

    alpha_n = alpha0 + n / 2
    beta_n = beta0 + 0.5 * (n * s2 + (kappa0 * n * (xbar - mu0) ** 2) / kappa_n)
    sigma2_n = beta_n / max(1e-9, (alpha_n - 1))

    return {"mean": float(mu_n), "var": float(sigma2_n / kappa_n)}

def prob_diff_greater(mu_a: float, var_a: float,
                      mu_b: float, var_b: float,
                      delta: float = 0.0) -> float:
    """
    Δ = μ_B - μ_A ~ Normal(μ_b-μ_a, var_a + var_b)
    """
    mean = mu_b - mu_a
    sd = math.sqrt(var_a + var_b + 1e-12)

    z = (mean - delta) / sd
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def analyze_ab_normal_from_df(df: pd.DataFrame, group_col: str, value_col: str,
                              priors: Optional[Dict[str, float]] = None,
                              out_dir: Optional[str] = None) -> Dict[str, float]:

    priors = priors or {"mu0": 0.0, "kappa0": 1e-6, "alpha0": 1e-6, "beta0": 1e-6}

    groups = sorted(list(df[group_col].unique()))
    print(groups)
    assert len(groups) == 2
    a, b = groups[0], groups[1]

    A = df[df[group_col] == a][value_col].astype(float)
    B = df[df[group_col] == b][value_col].astype(float)
    xa, xb = float(A.mean()), float(B.mean())
    s2a, s2b = float(A.var(ddof=1)), float(B.var(ddof=1))
    na, nb = int(A.shape[0]), int(B.shape[0])

    post_a = posterior_normal_unknown_var(priors["mu0"], priors["kappa0"], priors["alpha0"], priors["beta0"],
                                          xa, s2a, na)
    post_b = posterior_normal_unknown_var(priors["mu0"], priors["kappa0"], priors["alpha0"], priors["beta0"],
                                          xb, s2b, nb)

    p_sup = prob_diff_greater(post_a["mean"], post_a["var"], post_b["mean"], post_b["var"], delta=0.0)

    diff_mean = post_b["mean"] - post_a["mean"]
    diff_var = post_a["var"] + post_b["var"]
    diff_sd = math.sqrt(diff_var)

    loss_a = diff_sd * norm.pdf(diff_mean / diff_sd) + diff_mean * norm.cdf(diff_mean / diff_sd)
    loss_b = diff_sd * norm.pdf(-diff_mean / diff_sd) - diff_mean * norm.cdf(-diff_mean / diff_sd)

    res = {
        "post_a": post_a,
        "post_b": post_b,
        "p_superior_B_over_A": float(p_sup),
        "diff_mean": float(diff_mean),
        "diff_var": float(diff_var),
        "expected_loss_A": loss_a,
        "expected_loss_B": loss_b
    }
    
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/bayes_normal_report.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


    return res
from typing import Dict, List, Optional
import math
import json

def two_mean_ttest(mean_a: float, var_a: float, n_a: int,
                   mean_b: float, var_b: float, n_b: int,
                   equal_var: bool = False,
                   alternative: str = "two-sided") -> Dict[str, float]:
    """
    Returns: {'t','df','p_value','diff','se','ci95'}
    """
    diff = mean_b - mean_a
    if equal_var:
        # equal_var=True -> Student's T-test
        sp2 = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(1, n_a + n_b - 2)
        se = math.sqrt(sp2 * (1 / max(1, n_a) + 1 / max(1, n_b)) + 1e-12)
        df = n_a + n_b - 2    # (df, degree of freedom)

    else:    # equal_var=False -> Welch's T-test
        se = math.sqrt(var_a / max(1, n_a) + var_b / max(1, n_b) + 1e-12)
        # Welch–Satterthwaite
        num = (var_a / max(1, n_a) + var_b / max(1, n_b)) ** 2
        den = (var_a ** 2) / (max(1, n_a ** 2 * (n_a - 1))) + (var_b ** 2) / (max(1, n_b ** 2 * (n_b - 1)))
        df = num / (den + 1e-12)

    t = diff / se
    _phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    if alternative == "two-sided":
        p = 2 * (1 - _phi(abs(t)))
    elif alternative == "larger":
        p = 1 - _phi(t)
    else:
        p = _phi(t)

    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    return {
        "t": float(t), "df": float(df), "p_value": float(p), "diff": float(diff),
        "se": float(se), "ci95": (float(ci_low), float(ci_high))
    }



def analyze_ab_from_dataframe(df: pd.DataFrame, group_col: str, outcome_col: str,
                              out_dir: Optional[str] = None) -> Dict[str, float]:

    groups = sorted(list(df[group_col].unique()))
    print(groups)
    assert len(groups) == 2
    a, b = groups[0], groups[1]

    # group outcome
    d_a = df[df[group_col] == a][outcome_col]
    d_b = df[df[group_col] == b][outcome_col]

    # Welch t-test
    res = two_mean_ttest(
        float(d_a.mean()), float(d_a.var(ddof=1)), int(d_a.shape[0]),
        float(d_b.mean()), float(d_b.var(ddof=1)), int(d_b.shape[0]),
        equal_var=False
    )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/report.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    return res
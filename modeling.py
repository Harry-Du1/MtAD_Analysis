# mtad/modeling.py
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from . import config as C
from .io_utils import coerce_numeric, make_ad_binary

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def design_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = pd.get_dummies(df[cols].copy(), drop_first=True)
    for c in X.columns:
        X[c] = coerce_numeric(X[c])
    # standardize non-binary
    for c in X.columns:
        vals = X[c].dropna().unique()
        if set(np.unique(vals)).issubset({0.0, 1.0}):  # dummy
            continue
        mu, sd = np.nanmean(X[c].values), np.nanstd(X[c].values)
        if sd and sd > 0:
            X[c] = (X[c] - mu) / sd
    return X.dropna(axis=1, how="all").astype(float)

def _drop_zero_variance(X: pd.DataFrame) -> pd.DataFrame:
    nunique = X.nunique(dropna=False)
    keep = nunique[nunique > 1].index
    if "const" in X.columns and "const" not in keep:
        keep = keep.union(pd.Index(["const"]))
    return X[keep]

def _drop_separation_dummies(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    yb = y.astype(float)
    keep_cols = []
    for c in X.columns:
        if c == "const":
            keep_cols.append(c); continue
        xc = X[c]
        vals = xc.dropna().unique()
        if set(np.unique(vals)).issubset({0.0, 1.0}):
            m1 = ((xc == 1) & (yb == 1)).sum()
            m0 = ((xc == 1) & (yb == 0)).sum()
            if m1 == 0 or m0 == 0:
                continue
        keep_cols.append(c)
    return X[keep_cols]

def _clip_extremes(X: pd.DataFrame, zmax: float = 8.0) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        if c == "const": continue
        v = Xc[c].values
        if np.nanstd(v) > 0:
            m, s = np.nanmean(v), np.nanstd(v)
            Xc[c] = np.clip(v, m - zmax*s, m + zmax*s)
    return Xc

def fit_logit(y: pd.Series, X: pd.DataFrame):
    y = coerce_numeric(y)
    X = sm.add_constant(X, has_constant="add")
    valid = (~y.isna()) & X.notna().all(axis=1)
    if valid.sum() < C.MIN_SAMPLES_FIT:
        return None, valid
    yv = y.loc[valid].astype(float)
    Xv = X.loc[valid].astype(float)
    Xv = _drop_zero_variance(Xv)
    Xv = _drop_separation_dummies(yv, Xv)
    Xv = _clip_extremes(Xv, zmax=8.0)

    try:
        glm = sm.GLM(yv, Xv, family=sm.families.Binomial())
        glm_res = glm.fit(maxiter=300)
        glm_res = glm_res.get_robustcov_results(cov_type="HC3")
        return glm_res, valid
    except Exception:
        pass
    try:
        logit = sm.Logit(yv, Xv).fit(disp=False, maxiter=300, method="newton")
        logit = logit.get_robustcov_results(cov_type="HC3")
        return logit, valid
    except Exception:
        pass
    try:
        pen = None
        if "const" in Xv.columns:
            pen = np.ones(Xv.shape[1]); pen[list(Xv.columns).index("const")] = 0.0
        reg = sm.Logit(yv, Xv).fit_regularized(alpha=1.0, L1_wt=0.0, disp=False, penalization=pen)
        return reg, valid
    except Exception:
        return None, valid

def _fdr(df: pd.DataFrame, pcol="p", method="fdr_bh") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["p_adj"] = multipletests(df[pcol].fillna(1.0).values, method=method)[1]
    return df

# ----- Public APIs -----
def run_binary(df: pd.DataFrame, metab_cols: list, outcome_col: str,
               positive_labels: set, covariates: list, fdr_method="fdr_bh") -> pd.DataFrame:
    y = make_ad_binary(df[outcome_col], set(map(str, positive_labels)))
    covars_present = [c for c in covariates if c in df.columns]
    Xc = design_matrix(df, covars_present) if covars_present else pd.DataFrame(index=df.index)
    rows = []
    for m in metab_cols:
        X = Xc.copy()
        X["__metab__"] = coerce_numeric(df[m])
        res, valid = fit_logit(y, X)
        if res is None or "__metab__" not in getattr(res, "params", {}):
            continue
        b = float(res.params["__metab__"]); se = float(getattr(res, "bse", {}).get("__metab__", np.nan))
        p = float(getattr(res, "pvalues", {}).get("__metab__", np.nan))
        OR = float(np.exp(b))
        lo = float(np.exp(b - 1.96*se)) if np.isfinite(se) else np.nan
        hi = float(np.exp(b + 1.96*se)) if np.isfinite(se) else np.nan
        rows.append({"metabolite": m, "beta": b, "OR": OR, "OR_lo95": lo, "OR_hi95": hi, "p": p, "n": int(valid.sum())})
    return _fdr(pd.DataFrame(rows).sort_values("p"), "p", fdr_method)

def run_linear(df: pd.DataFrame, metab_cols: list, outcome_col: str,
               covariates: list, fdr_method: str = "fdr_bh") -> pd.DataFrame:
    """
    Per-metabolite linear regression using FWL residualization:
      1) ry = residuals of outcome ~ covariates
      2) rx = residuals of metabolite ~ covariates
      3) ry ~ zscore(rx)  (simple OLS with intercept)
    Drops metabolites whose residual SD is tiny (non-identifiable).
    """
    # thresholds to avoid numeric explosions (tune if needed)
    MIN_N = 8
    MIN_RX_SD = 1e-6   # drop if residual SD < this
    MIN_RY_SD = 1e-8   # drop if outcome residual SD < this

    rows, dropped = [], []

    # outcome & design for covariates
    y = pd.to_numeric(df[outcome_col], errors="coerce").astype(float)
    covars_present = [c for c in (covariates or []) if c in df.columns]
    Xc = design_matrix(df, covars_present) if covars_present else pd.DataFrame(index=df.index)

    # residualize outcome once
    if not Xc.empty:
        valid_y = (~y.isna()) & Xc.notna().all(axis=1)
        if valid_y.sum() < MIN_N:
            return pd.DataFrame(columns=["metabolite","beta","se","p","n","p_adj"])
        fit_y = sm.OLS(y.loc[valid_y], sm.add_constant(Xc.loc[valid_y], has_constant="add")).fit()
        ry_full = pd.Series(index=df.index, dtype=float)
        ry_full.loc[valid_y] = y.loc[valid_y] - fit_y.fittedvalues
    else:
        ry_full = y.copy()

    for m in metab_cols:
        xm = pd.to_numeric(df[m], errors="coerce").astype(float)

        valid = (~ry_full.isna()) & (~xm.isna())
        if not Xc.empty:
            valid &= Xc.notna().all(axis=1)
        n = int(valid.sum())
        if n < MIN_N:
            continue

        # residualize metabolite
        if not Xc.empty:
            fit_x = sm.OLS(xm.loc[valid], sm.add_constant(Xc.loc[valid], has_constant="add")).fit()
            rx = xm.loc[valid] - fit_x.fittedvalues
        else:
            rx = xm.loc[valid]

        ry = ry_full.loc[valid]

        # center & SD-floor checks (anti-infinite)
        ry = ry - ry.mean()
        ry_sd = float(np.nanstd(ry.values))
        if not np.isfinite(ry_sd) or ry_sd < MIN_RY_SD:
            dropped.append({"metabolite": m, "n": n, "reason": "outcome_residual_too_small"})
            continue

        rx = rx - rx.mean()
        rx_sd = float(np.nanstd(rx.values))
        if not np.isfinite(rx_sd) or rx_sd < MIN_RX_SD:
            dropped.append({"metabolite": m, "n": n, "reason": "metab_residual_too_small"})
            continue

        # z-score predictor residual to stabilize numerics
        rx_z = rx / rx_sd
        X = sm.add_constant(pd.DataFrame({"__metab__": rx_z}), has_constant="add")

        try:
            fit = sm.OLS(ry, X).fit()
            b  = float(fit.params["__metab__"])
            se = float(fit.bse["__metab__"])
            p  = float(fit.pvalues["__metab__"])
        except Exception:
            dropped.append({"metabolite": m, "n": n, "reason": "ols_failed"})
            continue

        if not (np.isfinite(b) and np.isfinite(se) and np.isfinite(p)):
            dropped.append({"metabolite": m, "n": n, "reason": "nonfinite_stats"})
            continue

        rows.append({"metabolite": m, "beta": b, "se": se, "p": p, "n": n})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_adj"] = multipletests(out["p"].values, method=fdr_method)[1]
        out = out.sort_values(["p_adj","p","metabolite"]).reset_index(drop=True)

    # QC report for transparency
    if dropped:
        q = pd.DataFrame(dropped)
        (C.OUTDIR / "QC").mkdir(parents=True, exist_ok=True)
        q.to_csv(C.OUTDIR / "QC" / f"dropped_{outcome_col}.csv", index=False)
        print(f"[linear] dropped {len(q)} metabolites for {outcome_col} "
              f"(see results/QC/dropped_{outcome_col}.csv)")

    return out

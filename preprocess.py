# mtad/preprocess.py
import numpy as np
import pandas as pd
from typing import List, Optional
from . import config as C
from .io_utils import coerce_numeric

def metabolite_call_rates(df: pd.DataFrame, metab_cols: List[str]) -> pd.Series:
    return df[metab_cols].notna().mean(axis=0)

def filter_by_callrate(df: pd.DataFrame, metab_cols: List[str], min_rate: float) -> List[str]:
    cr = metabolite_call_rates(df, metab_cols)
    keep = cr[cr >= min_rate].index.tolist()
    dropped = sorted(set(metab_cols) - set(keep))
    print(f"Call-rate filter: keeping {len(keep)}/{len(metab_cols)} (dropped {len(dropped)} < {min_rate:.0%})")
    pd.DataFrame({"CHEM_ID": cr.index, "call_rate": cr.values}).sort_values("call_rate") \
        .to_csv(C.OUTDIR / "QC_call_rates.csv", index=False)
    return keep

def _halfmin(vec: pd.Series) -> float:
    v = coerce_numeric(vec).values
    v = v[np.isfinite(v)]
    pos = v[v > 0]
    if pos.size: return float(np.nanmin(pos) / 2.0)
    return float(np.nanmin(v)) if v.size else 0.0

def impute_metabolites(df: pd.DataFrame, metab_cols: List[str],
                       strategy: str, batch_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    present_batches = [c for c in (batch_cols or []) if c in df.columns]
    if strategy.endswith("_by_batch") and present_batches:
        lbl = "__BATCH__"
        df[lbl] = df[present_batches].astype(str).agg("|".join, axis=1)
        for m in metab_cols:
            for _, idx in df.groupby(lbl).groups.items():
                col = df.loc[idx, m]
                fill = float(np.nanmedian(col.values)) if strategy == "median_by_batch" else _halfmin(col)
                df.loc[idx, m] = col.fillna(fill)
        df.drop(columns=[lbl], inplace=True)
    else:
        for m in metab_cols:
            fill = float(np.nanmedian(df[m].values)) if strategy == "median_global" else _halfmin(df[m])
            df[m] = df[m].fillna(fill)
    return df

def preprocess_metabolites(df: pd.DataFrame, metab_cols: list[str], log1p: bool = True) -> pd.DataFrame:
    df = df.copy()
    X = df[metab_cols].astype(float)

    if log1p:
        # keep as pandas: clip then log1p (pandas ufunc preserves DataFrame)
        X = X.clip(lower=-0.999999)
        X = np.log1p(X)

    mu = X.mean(axis=0)                 # Series (per-column mean)
    sd = X.std(axis=0, ddof=0)          # Series (per-column std)
    sd = sd.replace([np.inf, -np.inf], np.nan)
    sd = sd.where(sd >= 1e-6, other=1.0)  # floor tiny SDs to 1.0
    sd = sd.fillna(1.0)

    df[metab_cols] = (X - mu) / sd      # broadcast by column labels
    return df



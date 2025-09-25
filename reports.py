# mtad/reports.py
import pandas as pd
from typing import Dict, Optional
from . import config as C

def print_significant(res: pd.DataFrame, name_map: Optional[Dict[str, str]] = None,
                      alpha: float = 0.05, max_rows: Optional[int] = 120):
    if res.empty or "p_adj" not in res.columns:
        print("No results to report."); return
    sig = res.loc[res["p_adj"] < alpha].copy()
    if sig.empty:
        print(f"No metabolites significant at FDR < {alpha}."); return
    if name_map:
        sig["name"] = sig["metabolite"].map(name_map).fillna(sig["metabolite"])
    cols = (["metabolite"] + (["name"] if "name" in sig.columns else []) +
            ["OR", "OR_lo95", "OR_hi95", "beta", "p", "p_adj", "n"])
    sig = sig.sort_values(["p_adj", "p"], ascending=True)
    print(f"\nSignificant metabolites (FDR < {alpha}) — n={len(sig)}")
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", None,
                           "display.width", 160, "display.float_format", lambda x: f"{x:.3g}"):
        print(sig[cols].to_string(index=False))
    (C.OUTDIR / "significant").mkdir(parents=True, exist_ok=True)
    sig[cols].to_csv(C.OUTDIR / "significant" / "AD_logistic_significant.csv", index=False)

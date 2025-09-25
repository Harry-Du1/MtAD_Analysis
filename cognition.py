# mtad/cognition.py
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd

from . import config as C
from .modeling import run_linear
from .viz import HAVE_SNS
import matplotlib.pyplot as plt

if HAVE_SNS:
    import seaborn as sns

# Default cognition outcomes (present-only are analyzed)
COGNITION_LEVELS = [
    "cogn_global_lv", "cogn_ep_lv", "cogn_po_lv",
    "cogn_ps_lv", "cogn_se_lv", "cogn_wo_lv",
]
COGNITION_SLOPES = [
    "cognep_demog_slope", "cognep_path_slope",
    "cogng_demog_slope", "cogng_path_slope",
    "cognpo_demog_slope", "cognpo_path_slope",
    "cognps_demog_slope", "cognps_path_slope",
    "cognse_demog_slope", "cognse_path_slope",
    "cognwo_demog_slope", "cognwo_path_slope",
]

def _present(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def _linkage_matrix(results_by_outcome: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a matrix (rows=outcomes, cols=metabolites) with signed -log10(FDR)
    using sign(beta)*(-log10(p_adj)). Missing = 0.
    """
    mats = []
    all_mets = sorted({m for r in results_by_outcome.values() for m in r["metabolite"]})
    for outcome, res in results_by_outcome.items():
        tmp = res.set_index("metabolite")
        s = np.sign(tmp["beta"]).fillna(0.0)
        q = tmp["p_adj"].fillna(1.0)
        val = s * (-np.log10(q + 1e-300))
        row = pd.Series(0.0, index=all_mets, name=outcome)
        row.loc[val.index] = val
        mats.append(row)
    return pd.DataFrame(mats)

def _save_heatmap_linkage(M: pd.DataFrame, outpath: Path):
    if M.empty:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    # keep only strongest metabolites (by max |score| across outcomes)
    keep = M.abs().max(0).sort_values(ascending=False)
    keep_cols = keep.index[: max(10, min(40, len(keep)))]  # cap width
    Mplot = M.loc[:, keep_cols]

    if HAVE_SNS:
        g = sns.clustermap(
            Mplot, cmap="vlag", center=0, figsize=(10, 6),
            row_cluster=True, col_cluster=True
        )
        # remove ALL tick labels (your preference)
        g.ax_heatmap.set_xticklabels([]); g.ax_heatmap.set_yticklabels([])
        g.ax_heatmap.tick_params(left=False, bottom=False)
        g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
        g.savefig(outpath, dpi=200, bbox_inches="tight"); plt.close(g.fig)
    else:
        plt.figure(figsize=(10, 5))
        plt.imshow(Mplot.values, aspect="auto", cmap="coolwarm")
        plt.colorbar(label="signed -log10(FDR)")
        plt.xticks([]); plt.yticks([])
        plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def analyze_cognition(df: pd.DataFrame, metab_cols: List[str],
                      covariates: Optional[List[str]] = None,
                      outcomes: Optional[List[str]] = None,
                      fdr_method: str = "fdr_bh") -> Dict[str, pd.DataFrame]:
    """
    Run per-metabolite linear regressions against cognition outcomes.
    Saves one CSV per outcome and a linkage heatmap-ready matrix.
    Returns dict: outcome -> results dataframe.
    """
    covariates = covariates or C.COVARIATES
    out_dir = C.OUTDIR / "cognition"
    out_dir.mkdir(parents=True, exist_ok=True)

    if outcomes is None:
        outcomes = _present(df, COGNITION_LEVELS + COGNITION_SLOPES)
    if not outcomes:
        print("No cognition outcomes found."); return {}

    results_by_outcome: Dict[str, pd.DataFrame] = {}
    for y in outcomes:
        res = run_linear(df, metab_cols, outcome_col=y, covariates=covariates, fdr_method=fdr_method)
        res.to_csv(out_dir / f"{y}.csv", index=False)
        results_by_outcome[y] = res
        print(f"[cognition] {y}: n_mets={len(res)}  sig(FDR<0.05)={(res['p_adj']<0.05).sum()}")

    # Build & save linkage matrix
    M = _linkage_matrix(results_by_outcome)
    M.to_csv(out_dir / "linkage_matrix_signed_log10FDR.csv")
    _save_heatmap_linkage(M, out_dir / "linkage_heatmap.png")

    return results_by_outcome

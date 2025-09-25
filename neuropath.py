# mtad/neuropath.py
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd

from . import config as C
from .modeling import run_linear
from .viz import HAVE_SNS
import matplotlib.pyplot as plt

if HAVE_SNS:
    import seaborn as sns

# Core neuropathology measures; we'll use what's present
PATHOLOGY_CORE = [
    "braaksc", "ceradsc", "amyloid", "plaq_d", "plaq_n", "nft", "tangles"
]

# Optional regionals (used in individual analyses if present; not in composite by default)
PATHOLOGY_REGIONAL = [
    "plaq_d_ag","plaq_d_ec","plaq_d_hip","plaq_d_mf","plaq_d_mt",
    "plaq_n_ag","plaq_n_ec","plaq_n_hip","plaq_n_mf","plaq_n_mt",
    "nft_ag","nft_ec","nft_hip","nft_mf","nft_mt"
]

def _present(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def build_pathology_index(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """
    Composite 'pathology_index' = mean z across core pathology columns present.
    Higher = worse pathology.
    """
    cols = _present(df, PATHOLOGY_CORE)
    if not cols:
        raise ValueError("No core pathology columns found to build a composite index.")
    X = df[cols].astype(float)
    Z = (X - X.mean(0)) / X.std(0, ddof=0).replace(0, 1)
    idx = Z.mean(1)  # simple average of z-scores
    idx.name = "pathology_index"
    return idx, cols

def _linkage_matrix(results_by_outcome: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    keep = M.abs().max(0).sort_values(ascending=False)
    keep_cols = keep.index[: max(10, min(40, len(keep)))]
    Mplot = M.loc[:, keep_cols]

    if HAVE_SNS:
        g = sns.clustermap(Mplot, cmap="vlag", center=0, figsize=(10, 6),
                           row_cluster=True, col_cluster=True)
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

def analyze_neuropathology(df: pd.DataFrame, metab_cols: List[str],
                           covariates: Optional[List[str]] = None,
                           include_individual_outcomes: bool = True,
                           fdr_method: str = "fdr_bh") -> Dict[str, pd.DataFrame]:
    """
    Regress metabolites against:
      (A) a composite pathology index (mean z across core measures present)
      (B) individual pathology outcomes (optional)
    Saves CSVs and a linkage matrix + heatmap.
    Returns dict: outcome -> results dataframe.
    """
    covariates = covariates or C.COVARIATES
    out_dir = C.OUTDIR / "neuropath"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (A) Composite index
    idx, used_cols = build_pathology_index(df)
    df_idx = df.copy()
    df_idx["pathology_index"] = idx
    res_index = run_linear(df_idx, metab_cols, outcome_col="pathology_index",
                           covariates=covariates, fdr_method=fdr_method)
    res_index.to_csv(out_dir / "pathology_index.csv", index=False)
    print(f"[neuropath] pathology_index (from {used_cols})  sig(FDR<0.05)={(res_index['p_adj']<0.05).sum()}")

    results_by_outcome: Dict[str, pd.DataFrame] = {"pathology_index": res_index}

    # (B) Individual outcomes
    if include_individual_outcomes:
        indiv = _present(df, PATHOLOGY_CORE + PATHOLOGY_REGIONAL)
        for y in indiv:
            res = run_linear(df, metab_cols, outcome_col=y, covariates=covariates, fdr_method=fdr_method)
            res.to_csv(out_dir / f"{y}.csv", index=False)
            results_by_outcome[y] = res
            print(f"[neuropath] {y}: n_mets={len(res)}  sig(FDR<0.05)={(res['p_adj']<0.05).sum()}")

    # Linkage matrix across outcomes
    M = _linkage_matrix(results_by_outcome)
    M.to_csv(out_dir / "linkage_matrix_signed_log10FDR.csv")
    _save_heatmap_linkage(M, out_dir / "linkage_heatmap.png")

    return results_by_outcome

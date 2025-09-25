# mtad/viz.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns
HAVE_SNS = True
from . import config as C
from matplotlib.patches import Patch
from matplotlib.patches import Patch

def volcano_plot(df_res: pd.DataFrame, outpath: Path,
                 name_map: Optional[Dict[str, str]] = None,
                 label_top: int = 15, use_padj: bool = True):
    if df_res.empty: 
        return

    metric = "p_adj" if use_padj else "p"

    x = np.log2(df_res["OR"].astype(float).replace([np.inf, -np.inf], np.nan))
    y = -np.log10(df_res[metric].astype(float).fillna(1.0) + 1e-300)
    signif = df_res["p_adj"].values < 0.05

    plt.figure(figsize=(7.6, 5.6))
    plt.scatter(x[~signif], y[~signif], s=16, alpha=0.7)
    plt.scatter(x[signif], y[signif], s=20, alpha=0.9)
    plt.axhline(-np.log10(0.05), linestyle="--", linewidth=1)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("log2(OR) per 1 SD"); plt.ylabel("-log10(adj-p)" if use_padj else "-log10(p)")
    plt.title("AD vs nonAD (per-metabolite logistic)")

    # --- Top-2 by chosen p-value metric ---
    top2 = df_res.nsmallest(2, metric)

    # Print their names (mapped if available)
    names_to_print = []
    for _, r in top2.iterrows():
        lab = name_map.get(r["metabolite"], r["metabolite"]) if name_map else r["metabolite"]
        names_to_print.append(str(lab))
    print("[volcano] Top 2 metabolites:", "; ".join(names_to_print))

    # Annotate only these two
    for _, r in top2.iterrows():
        lab = name_map.get(r["metabolite"], r["metabolite"]) if name_map else r["metabolite"]
        tx = np.log2(r["OR"]) if np.isfinite(r["OR"]) and r["OR"] > 0 else 0.0
        ty = -np.log10(r[metric] + 1e-300)
        plt.text(tx, ty, str(lab), fontsize=8)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()


def heatmap_top_hits(df_all: pd.DataFrame, metab_cols: List[str],
                     df_res: pd.DataFrame, outpath: Path,
                     top_n: int = C.TOP_N_FOR_HEATMAP,
                     name_map: Optional[Dict[str, str]] = None,
                     label_metabolites: bool = True,
                     label_max: int = 40,
                     rotate_deg: int = 80):
    if df_res.empty:
        return

    chosen = (df_res[df_res["p_adj"] < 0.05].nsmallest(top_n, "p_adj")
              if (df_res["p_adj"] < 0.05).any() else df_res.nsmallest(top_n, "p_adj"))
    cols = [m for m in chosen["metabolite"].tolist() if m in metab_cols]
    if not cols:
        return

    id_col = (C.ID_COL_META if C.ID_COL_META in df_all.columns else
              (getattr(C, "ID_COL_METAB", None) if getattr(C, "ID_COL_METAB", None) in df_all.columns else "individualID"))

    mat = df_all.set_index(id_col)[cols].copy()
    if name_map:
        mat.columns = [name_map.get(c, c) for c in mat.columns]

    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=0).replace(0, 1)
    mat = (mat - mu) / sd

    ad_raw = df_all.set_index(id_col)[C.AD_COLUMN].astype(str).reindex(mat.index)
    ad_label = ad_raw.apply(lambda v: "AD" if v in C.AD_POSITIVE_LABELS else "non-AD")

    outpath.parent.mkdir(parents=True, exist_ok=True)

    if HAVE_SNS:
        import seaborn as sns
        from matplotlib.patches import Patch

        pal = {"AD": "#d62728", "non-AD": "#1f77b4"}
        row_colors = ad_label.map(pal)

        g = sns.clustermap(
            mat.fillna(0.0), method="average", metric="euclidean",
            row_cluster=True, col_cluster=True,
            row_colors=row_colors,
            cmap="vlag", center=0, figsize=(10, 8)
        )

        # row ticks off
        g.ax_heatmap.set_yticklabels([])
        g.ax_heatmap.tick_params(left=False)

        # column labels only if manageable
        if label_metabolites and mat.shape[1] <= label_max:
            labels = list(getattr(g, "data2d", mat).columns)
            g.ax_heatmap.set_xticklabels(labels, rotation=rotate_deg, ha="right", fontsize=8)
            g.ax_heatmap.tick_params(bottom=True)
        else:
            g.ax_heatmap.set_xticklabels([])
            g.ax_heatmap.tick_params(bottom=False)

        g.ax_heatmap.set_xlabel("Metabolites")
        g.ax_heatmap.set_ylabel("Individual")

        handles = [Patch(facecolor=pal[k], label=k) for k in ("AD", "non-AD")]
        g.ax_heatmap.legend(handles=handles, title=C.AD_COLUMN, loc="upper left",
                            bbox_to_anchor=(1.02, 1.0), frameon=False)

        g.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(g.fig)




def _zscore_cols(mat: pd.DataFrame) -> pd.DataFrame:
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=0).replace(0, 1)
    return (mat - mu) / sd

def heatmap_all_metabolites(
    df_all: pd.DataFrame,
    metab_cols: list[str],
    outpath: Path,
    name_map: dict[str, str] | None = None,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    standardize: bool = True,
    figsize: tuple[float, float] = (12, 8),
):
    """
    Draw a heatmap with *all* metabolites in `metab_cols`.
    - No tick labels (crowded); but axis labels are present.
    - Row color bar indicates AD vs non-AD.
    - Columns are z-scored for display only.
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # 1) Matrix: rows = individuals, cols = metabolites
    if C.ID_COL_METAB not in df_all.columns:
        id_col = C.ID_COL_META if C.ID_COL_META in df_all.columns else "individualID"
    else:
        id_col = C.ID_COL_METAB
    mat = df_all.set_index(id_col)[metab_cols].copy()

    # 2) Pretty column names (optional)
    if name_map:
        mat.columns = [name_map.get(c, c) for c in mat.columns]

    # 3) Standardize columns for display
    if standardize:
        mat = _zscore_cols(mat)

    # 4) Row colors = AD vs non-AD
    ad_raw = df_all.set_index(id_col)[C.AD_COLUMN].astype(str).reindex(mat.index)
    ad_label = ad_raw.apply(lambda v: "AD" if v in C.AD_POSITIVE_LABELS else "non-AD")
    if HAVE_SNS:
        pal = {"AD": "#d62728", "non-AD": "#1f77b4"}  # red / blue
        row_colors = ad_label.map(pal)

        g = sns.clustermap(
            mat.fillna(0.0),
            method="average", metric="euclidean",
            row_cluster=cluster_rows, col_cluster=cluster_cols,
            row_colors=row_colors,
            cmap="vlag", center=0,
            figsize=figsize
        )
        # No ID ticks, but keep axes labels
        g.ax_heatmap.set_xticklabels([]); g.ax_heatmap.set_yticklabels([])
        g.ax_heatmap.tick_params(left=False, bottom=False)
        g.ax_heatmap.set_xlabel("Metabolites")
        g.ax_heatmap.set_ylabel("Individual")

        # Legend for AD/non-AD
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=pal[k], label=k) for k in ("AD", "non-AD")]
        g.ax_heatmap.legend(handles=handles, title=C.AD_COLUMN, loc="upper left",
                            bbox_to_anchor=(1.02, 1.0), frameon=False)

        g.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
    else:
        # Fallback without seaborn: simple heatmap + side color strip
        fig = plt.figure(figsize=figsize)
        # main heatmap
        ax = fig.add_axes([0.08, 0.1, 0.75, 0.8])
        im = ax.imshow(mat.fillna(0.0).values, aspect="auto", cmap="coolwarm")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Metabolites"); ax.set_ylabel("Individual")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("z-score")

        # side color bar for AD/non-AD
        axc = fig.add_axes([0.85, 0.1, 0.015, 0.8])
        pal = {"AD": (0.84, 0.16, 0.15), "non-AD": (0.12, 0.47, 0.71)}
        color_col = ad_label.map(pal).apply(lambda rgb: list(rgb)).values
        color_img = np.array(color_col, dtype=float).reshape(-1, 1, 3)
        axc.imshow(color_img, aspect="auto")
        axc.set_xticks([]); axc.set_yticks([])
        axc.set_title("AD", fontsize=9, pad=6)

        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)


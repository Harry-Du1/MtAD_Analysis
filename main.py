# MtAD_Analysis/main.py

# ---- robust imports: works as package or as script ----
try:
    from . import config as C
    from .io_utils import load_and_merge, build_name_map
    from .preprocess import filter_by_callrate, impute_metabolites, preprocess_metabolites
    from .modeling import run_binary, run_linear
    from .viz import volcano_plot, heatmap_top_hits, heatmap_all_metabolites
    from .reports import print_significant
    try:
        from .covars import resolve_covariates
    except Exception:
        resolve_covariates = None
    from .cognition import analyze_cognition
    from .neuropath import analyze_neuropathology
except ImportError:
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    import config as C
    from io_utils import load_and_merge, build_name_map
    from preprocess import filter_by_callrate, impute_metabolites, preprocess_metabolites
    from modeling import run_binary, run_linear
    from viz import volcano_plot, heatmap_top_hits
    from reports import print_significant
    try:
        from covars import resolve_covariates
    except Exception:
        resolve_covariates = None
    from cognition import analyze_cognition
    from neuropath import analyze_neuropathology

import argparse
import numpy as np
from pathlib import Path

# ---------------- Shared helpers ----------------
def _load_and_prep(use_mito_only=None, min_callrate=None, impute_strategy=None):
    """Load -> select cols -> call-rate filter -> impute -> transform."""
    if use_mito_only is None:
        use_mito_only = C.USE_MITO_ONLY
    if min_callrate is None:
        min_callrate = C.MIN_METAB_CALLRATE
    if impute_strategy is None:
        impute_strategy = C.IMPUTE_STRATEGY

    df, metab_cols, _ = load_and_merge(C.MITO_FILE, C.META_FILE, C.METAB_FILE,
                                       use_mito_only=use_mito_only)
    print(f"[load] metabolite columns before filtering: {len(metab_cols)}")
    metab_cols = filter_by_callrate(df, metab_cols, min_rate=min_callrate)
    df = impute_metabolites(df, metab_cols, strategy=impute_strategy, batch_cols=C.BATCH_COLS_ALL)
    df = preprocess_metabolites(df, metab_cols, log1p=C.LOG1P_METABS)
    print(f"[prep] metabolite columns after prep: {len(metab_cols)}")
    return df, metab_cols

def _resolve_covs(df, args):
    """Resolve covariates from CLI or config; only keep columns present."""
    if resolve_covariates is None:
        # Fallback: use config covariates present in df
        covs = [c for c in C.COVARIATES if c in df.columns]
        print(f"[covars] using (fallback): {covs}")
        return covs
    covs = resolve_covariates(
        df,
        base=C.COVARIATES,
        preset=args.covars_preset,
        override_csv=args.covars,
        add_csv=args.add_covars,
        drop_csv=args.drop_covars,
        require_present=True,
    )
    print(f"[covars] using: {covs}")
    return covs

def _heatmap(df, metab_cols, res, pretty_map_with_id, top_n, tag):
    out = C.OUTDIR / (f"heatmap_{tag}.png")
    # cap top_n to number of available metabolites
    top_n = max(1, min(top_n, len(metab_cols)))
    print(f"[heatmap] plotting up to top_n={top_n} metabolites (of {len(metab_cols)} available)")
    heatmap_top_hits(df, metab_cols, res, out, top_n=top_n, name_map=pretty_map_with_id)
    print(f"[heatmap] saved → {out}")

# ---------------- Analyses ----------------
def run_ad_vs_nonad(args):
    df, metab_cols = _load_and_prep(
        use_mito_only=True,
        min_callrate=args.min_callrate,
        impute_strategy=args.impute_strategy
    )
    covs = _resolve_covs(df, args)

    res = run_binary(df, metab_cols, outcome_col=C.AD_COLUMN,
                     positive_labels=C.AD_POSITIVE_LABELS, covariates=covs)
    out = C.OUTDIR / ("AD_logistic_results_mito.csv" if args.mito_only else "AD_logistic_results_all.csv")
    res.to_csv(out, index=False)
    print(f"[ad] saved results: {out}  rows={len(res)}  sig(FDR<0.05)={(res['p_adj']<0.05).sum()}")

    pretty_map         = build_name_map(C.MITO_FILE, append_id=False)
    pretty_map_with_id = build_name_map(C.MITO_FILE, append_id=True)

    print_significant(res, name_map=pretty_map, alpha=0.05)

    vol = C.OUTDIR / ("volcano_mito.png" if args.mito_only else "volcano_all.png")
    volcano_plot(res, vol, name_map=pretty_map_with_id)
    print(f"[volcano] saved → {vol}")

    if args.heatmap_mode == "all":
        hp = C.OUTDIR / (f"heatmap_all_{'mito' if args.mito_only else 'all'}.png")
        print(f"[heatmap] plotting ALL metabolites: {len(metab_cols)} columns")
        heatmap_all_metabolites(df, metab_cols, hp, name_map=pretty_map_with_id)
    else:
        _heatmap(df, metab_cols, res, pretty_map_with_id, args.top_n, "mito" if args.mito_only else "all")


def run_linear_one(args):
    if not args.outcome:
        raise SystemExit("--outcome is required for 'linear'")
    df, metab_cols = _load_and_prep(
        use_mito_only=args.mito_only,
        min_callrate=args.min_callrate,
        impute_strategy=args.impute_strategy
    )
    covs = _resolve_covs(df, args)
    res = run_linear(df, metab_cols, outcome_col=args.outcome, covariates=covs)
    out = C.OUTDIR / f"linear_{args.outcome}_{'mito' if args.mito_only else 'all'}.csv"
    res.to_csv(out, index=False)
    print(f"[linear] saved results: {out}  rows={len(res)}  sig(FDR<0.05)={(res['p_adj']<0.05).sum()}")

def run_cognition(args):
    df, metab_cols = _load_and_prep(
        use_mito_only=True,
        min_callrate=args.min_callrate,
        impute_strategy=args.impute_strategy
    )
    covs = _resolve_covs(df, args)
    analyze_cognition(df, metab_cols, covariates=covs)

def run_pathology(args):
    df, metab_cols = _load_and_prep(
        use_mito_only=True,
        min_callrate=args.min_callrate,
        impute_strategy=args.impute_strategy
    )
    covs = _resolve_covs(df, args)
    analyze_neuropathology(df, metab_cols, covariates=covs,
                           include_individual_outcomes=getattr(args, "individual", False))

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="mtAD metabolite analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Global knobs
    parser.add_argument("--mito-only", action="store_true",
                        help="Restrict to CHEM_IDs in mitochondrial_metabolites.csv")
    parser.add_argument("--top-n", type=int, default=C.TOP_N_FOR_HEATMAP,
                        help="Max metabolites to display in heatmap")
    parser.add_argument("--min-callrate", type=float, default=C.MIN_METAB_CALLRATE,
                        help="Minimum non-missing fraction per metabolite (call-rate filter)")
    parser.add_argument("--impute-strategy", default=C.IMPUTE_STRATEGY,
                        choices=["median_by_batch","halfmin_by_batch","median_global"],
                        help="Imputation method for metabolites")

    # Covariate selection
    parser.add_argument("--covars", default=None,
                        help="Comma-separated covariates to use (overrides presets & defaults)")
    parser.add_argument("--covars-preset", default="batch",
                        choices=["minimal","demog","demog+batch","full", "batch"],
                        help="Choose a covariate preset")
    parser.add_argument("--add-covars", default=None,
                        help="Comma-separated covariates to add on top of chosen set")
    parser.add_argument("--drop-covars", default=None,
                        help="Comma-separated covariates to remove from chosen set")
    parser.add_argument("--heatmap-mode", choices=["top", "all"], default="top",
                    help="Plot top-N metabolites (top) or all metabolites (all)")


    # Subcommands
    sub.add_parser("ad", help="AD vs non-AD logistic per metabolite")

    p_lin = sub.add_parser("linear", help="Linear regression per metabolite against a numeric outcome")
    p_lin.add_argument("--outcome", required=True, help="Outcome column (e.g., cogn_global_lv, plaq_d)")

    sub.add_parser("cognition", help="Linear per metabolite for cognition levels + slopes")

    p_path = sub.add_parser("pathology", help="Linear per metabolite for neuropathology (composite + optional individual)")
    p_path.add_argument("--individual", action="store_true",
                        help="Also run individual pathology outcomes (Braak, CERAD, plaques, NFT, regionals)")

    args = parser.parse_args()

    np.random.seed(C.RANDOM_SEED)
    Path(C.OUTDIR).mkdir(parents=True, exist_ok=True)

    if args.cmd == "ad":
        run_ad_vs_nonad(args)
    elif args.cmd == "linear":
        run_linear_one(args)
    elif args.cmd == "cognition":
        run_cognition(args)
    elif args.cmd == "pathology":
        run_pathology(args)

if __name__ == "__main__":
    main()

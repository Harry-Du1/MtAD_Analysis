# mtad/io_utils.py
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config as C

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        if p.suffix.lower() in {".tsv", ".txt"}:
            return pd.read_csv(p, sep="\t", dtype=str)
        return pd.read_csv(p, sep=",", dtype=str)
    except Exception:
        return pd.read_csv(p, sep=None, engine="python", dtype=str)

def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def build_name_map(mito_path: str,
                   extra_catalog_path: Optional[str] = None,
                   append_id: bool = True) -> Dict[str, str]:
    """Map CHEM_ID → pretty label using mito file (and optional broader catalog)."""
    def _load(path: Optional[str]) -> pd.DataFrame:
        if not path: return pd.DataFrame()
        df = read_table(path)
        if df.empty: return df
        lower = {c.lower(): c for c in df.columns}
        def pick(*cands):
            for c in cands:
                if c in lower: return lower[c]
            return None
        idc = pick("chem_id","chemid","chem id","id")
        sn  = pick("short_name","shortname","short")
        nm  = pick("chemical_name","name")
        if not idc: return pd.DataFrame()
        keep = [idc] + [c for c in (sn, nm) if c]
        out = df[keep].copy()
        out.columns = ["CHEM_ID"] + (["SHORT_NAME"] if sn else []) + (["CHEMICAL_NAME"] if nm else [])
        out["CHEM_ID"] = out["CHEM_ID"].astype(str)
        return out

    cat = pd.concat([d for d in (_load(mito_path), _load(extra_catalog_path)) if not d.empty],
                    ignore_index=True) if (mito_path or extra_catalog_path) else pd.DataFrame()
    if cat.empty or "CHEM_ID" not in cat.columns:
        return {}
    if "SHORT_NAME" in cat.columns:
        cat["LABEL"] = cat["SHORT_NAME"]
    elif "CHEMICAL_NAME" in cat.columns:
        cat["LABEL"] = cat["CHEMICAL_NAME"]
    else:
        cat["LABEL"] = cat["CHEM_ID"]
    cat["LABEL"] = cat["LABEL"].astype(str).str.strip()
    cat = cat.dropna(subset=["CHEM_ID"]).drop_duplicates(subset=["CHEM_ID"])

    # make labels unique
    dup = cat["LABEL"].duplicated(keep=False)
    if dup.any():
        cat.loc[dup, "LABEL"] = (
            cat.loc[dup].groupby("LABEL").cumcount().radd(1).astype(str)
            .str.cat(cat.loc[dup,"LABEL"], sep=" (").add(")")
        )
    if append_id:
        cat["LABEL"] = cat["LABEL"] + " [" + cat["CHEM_ID"].astype(str) + "]"
    return dict(zip(cat["CHEM_ID"], cat["LABEL"]))

def make_ad_binary(series: pd.Series, positive_labels: set) -> pd.Series:
    s = series.astype(str).str.strip()
    return coerce_numeric(s.apply(lambda v: 1 if v in positive_labels else (0 if v != "nan" else np.nan)))

def load_and_merge(mito_path: str, meta_path: str, metab_path: str,
                   use_mito_only: bool) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Read metadata + wide metabolite matrix; select CHEM_ID columns; merge on individualID."""
    meta = read_table(meta_path)
    if C.ID_COL_META not in meta.columns:
        raise ValueError(f"Metadata must contain '{C.ID_COL_META}'.")
    meta[C.ID_COL_META] = meta[C.ID_COL_META].astype(str)

    metab = read_table(metab_path)
    if C.ID_COL_METAB not in metab.columns:
        raise ValueError(f"Metabolomics matrix must contain '{C.ID_COL_METAB}'.")
    metab[C.ID_COL_METAB] = metab[C.ID_COL_METAB].astype(str)
    metab.columns = [str(c) for c in metab.columns]

    # CHEM_ID-like (digits only)
    all_cols = [c for c in metab.columns if c != C.ID_COL_METAB]
    chemid_like = [c for c in all_cols if re.fullmatch(C.CHEMID_PATTERN, c)]

    chemid_to_name: Dict[str, str] = {}
    if use_mito_only:
        mito = read_table(mito_path)
        if "CHEM_ID" not in mito.columns:
            raise ValueError("mitochondrial_metabolites.csv must contain 'CHEM_ID'.")
        mito_ids = set(mito["CHEM_ID"].astype(str))
        keep_ids = [c for c in chemid_like if c in mito_ids]
        if "CHEMICAL_NAME" in mito.columns:
            chemid_to_name = dict(zip(mito["CHEM_ID"].astype(str), mito["CHEMICAL_NAME"].astype(str)))
        missing = sorted(mito_ids - set(keep_ids))
        if missing:
            C.OUTDIR.mkdir(parents=True, exist_ok=True)
            pd.Series(missing, name="missing_mito_CHEM_ID").to_csv(C.OUTDIR / "QC_missing_mito_ids.csv", index=False)
            print(f"NOTE: {len(missing)} mito CHEM_ID(s) not found. See results/QC_missing_mito_ids.csv")
    else:
        keep_ids = chemid_like

    metab = metab[[C.ID_COL_METAB] + keep_ids].copy()
    for c in keep_ids:
        metab[c] = coerce_numeric(metab[c])

    df = pd.merge(meta, metab, left_on=C.ID_COL_META, right_on=C.ID_COL_METAB, how="inner")
    if C.ID_COL_METAB != C.ID_COL_META:
        df.drop(columns=[C.ID_COL_METAB], inplace=True)

    # QC manifests
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.Series(keep_ids, name="CHEM_ID").to_csv(C.OUTDIR / "QC_metabolite_columns_used.csv", index=False)
    pd.Series(sorted(set(all_cols) - set(keep_ids)), name="non_metabolite_cols").to_csv(
        C.OUTDIR / "QC_non_metabolite_columns.csv", index=False
    )

    return df, keep_ids, chemid_to_name

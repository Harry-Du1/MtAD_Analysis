# mtad/sets.py
import pandas as pd
from typing import Dict, List
from .io_utils import read_table
from . import config as C
from .modeling import run_binary, run_linear

def build_metabolite_sets(catalog_path: str, level: str = "SUB_PATHWAY") -> Dict[str, List[str]]:
    """Return set_name -> list of CHEM_IDs using catalog file columns CHEM_ID + level."""
    cat = read_table(catalog_path)
    if cat.empty or not {"CHEM_ID", level}.issubset(set(cat.columns)):
        return {}
    cat = cat.dropna(subset=["CHEM_ID", level]).copy()
    cat["CHEM_ID"] = cat["CHEM_ID"].astype(str)
    sets = (cat.groupby(level)["CHEM_ID"].apply(lambda s: sorted(set(map(str, s)))).to_dict())
    return sets

def score_sets(df: pd.DataFrame, metab_cols: List[str], set_map: Dict[str, List[str]], how: str = "mean") -> pd.DataFrame:
    """Compute per-sample activity scores for each metabolite set (mean z across members present)."""
    X = df[metab_cols].copy()
    scores = {}
    for sname, ids in set_map.items():
        members = [m for m in ids if m in X.columns]
        if len(members) < 2:  # skip tiny sets
            continue
        scores[sname] = X[members].mean(axis=1)
    S = pd.DataFrame(scores, index=df.index)
    S.insert(0, C.ID_COL_META, df[C.ID_COL_META])
    return S

# thin wrappers reusing per-feature APIs
def run_binary_sets(df_scores: pd.DataFrame, set_cols: list, outcome_col: str, positive_labels, covariates):
    return run_binary(df_scores, set_cols, outcome_col, positive_labels, covariates)

def run_linear_sets(df_scores: pd.DataFrame, set_cols: list, outcome_col: str, covariates):
    return run_linear(df_scores, set_cols, outcome_col, covariates)

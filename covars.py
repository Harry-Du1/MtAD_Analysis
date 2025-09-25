# MtAD_Analysis/covars.py
from typing import List, Optional
import pandas as pd
from . import config as C

COVARIATE_PRESETS = {
    "none":         [],
    "minimal":      ["age_death", "msex"],
    "demog":        ["age_death", "msex", "educ", "race"],
    "demog+batch":  ["age_death", "msex", "educ", "race",
                     "BATCH_209","BATCH_305","BATCH_400","BATCH_402"],
    "full":         ["age_death", "msex", "educ", "race",
                     "BATCH_209","BATCH_305","BATCH_400","BATCH_402",
                     # add more if you like, e.g. "apoe_genotype"
                    ],
    "batch":        ["BATCH_209","BATCH_305","BATCH_400","BATCH_402"]
}

def _parse_csv_list(s: Optional[str]) -> List[str]:
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def resolve_covariates(df: pd.DataFrame,
                       base: Optional[List[str]] = None,
                       preset: Optional[str] = None,
                       override_csv: Optional[str] = None,
                       add_csv: Optional[str] = None,
                       drop_csv: Optional[str] = None,
                       require_present: bool = True) -> List[str]:
    """
    Choose covariates using a preset or overrides.
    Priority (highest first): override_csv > (base or preset) then +add -drop.
    Returns only columns present in df if require_present=True.
    """
    if override_csv:
        covs = _parse_csv_list(override_csv)
    else:
        if preset:
            covs = list(COVARIATE_PRESETS.get(preset, []))
        else:
            covs = list(base if base is not None else C.COVARIATES)

    # apply add/drop
    add  = set(_parse_csv_list(add_csv))
    drop = set(_parse_csv_list(drop_csv))
    covs = [c for c in covs if c not in drop] + [c for c in add if c not in covs]

    if require_present:
        covs = [c for c in covs if c in df.columns]
    return covs

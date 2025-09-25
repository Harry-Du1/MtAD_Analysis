# mtad/config.py
from pathlib import Path

# --------- I/O ---------
MITO_FILE  = "mitochondrial_metabolites.csv"  # must have CHEM_ID; CHEMICAL_NAME optional
META_FILE  = "metadata/kellis_dejager_meta.dataset_1369_basic_03-13-2024.csv"
METAB_FILE = "metabolites/ROSMAP_Metabolon_HD4_Brain514_assay_data.csv"

ID_COL_META  = "individualID"
ID_COL_METAB = "individualID"

# --------- Analysis switches ---------
USE_MITO_ONLY     = True      # True → restrict to CHEM_IDs in MITO_FILE
LOG1P_METABS      = True        # log1p before z-score
MIN_SAMPLES_FIT   = 10
TOP_N_FOR_HEATMAP = 25

# AD label
AD_COLUMN          = "AD2status"
AD_POSITIVE_LABELS = {"AD"}

# Covariates (present columns only are used)
COVARIATES = [
    "age_death", "msex", "educ", "race",
    "BATCH_209", "BATCH_305", "BATCH_400", "BATCH_402",
    # "apoe_genotype",
]

# Missingness handling
MIN_METAB_CALLRATE = 0.70
IMPUTE_STRATEGY    = "median_by_batch"  # "median_by_batch" | "halfmin_by_batch" | "median_global"
BATCH_COLS_ALL     = ["BATCH_209","BATCH_305","BATCH_400","BATCH_402"]

# Patterns
CHEMID_PATTERN = r"^\d+$"  # CHEM_ID columns are digit-only strings: "35","41",...

# Misc
RANDOM_SEED = 1337
OUTDIR = Path("results")

"""Central configuration: paths, seeds, column groups and business costs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "default_of_credit_card_clients.xls"
DATA_URL = (
    "https://raw.githubusercontent.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/"
    "main/dataset/default%20of%20credit%20card%20clients.xls"
)
DATA_SHA256 = "c76095b2a65c6e1b0ef6bdfab785346065b02f3f50cab6751d5a46b4c01c9e44"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MLRUNS_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
TARGET = "default"

PAY_STATUS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]

# SEX is a protected attribute. It is kept in the data only to audit fairness
# and is never used as a model input.
PROTECTED = "SEX"

# Business cost assumption (documented in README): missing a defaulter loses
# roughly the exposure; wrongly declining a good customer loses the margin.
# Expressed as relative units, FN is 5x more expensive than FP.
COST_FN = 5.0
COST_FP = 1.0

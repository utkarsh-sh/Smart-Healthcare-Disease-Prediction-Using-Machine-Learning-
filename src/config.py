"""Central configuration for the BT60 Smart Healthcare Disease Prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
DB_PATH = ROOT / "data" / "predictions.db"

# Experimental protocol (fixed up-front so results are reproducible)
RANDOM_STATE = 42
TEST_SIZE = 0.20          # stratified hold-out, never used for model selection
CV_FOLDS = 5
CV_REPEATS = 3
SELECTION_CV_SEED = 7     # different folds for the final model-selection round
STABILITY_SEEDS = list(range(10))
BOOTSTRAP_RESAMPLES = 1000
SCREENING_TARGET_RECALL = 0.90

MODEL_VERSION = "1.0.0"
DISCLAIMER = (
    "Educational prototype built for an academic capstone project. Outputs are statistical "
    "risk estimates from public research datasets and are NOT a medical diagnosis. "
    "Consult a qualified clinician."
)

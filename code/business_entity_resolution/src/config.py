from pathlib import Path

# Base Paths (relative to repository root)
SRC_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SRC_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Learned artifacts (translit map, model, thresholds) ship with the code.
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
# Large intermediate caches (normalized parquet, candidates, features).
WORK_DIR = PROJECT_ROOT / "work"

# Output files
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

RANDOM_SEED = 42
BETA = 0.5  # F_beta evaluation metric (precision-focused)

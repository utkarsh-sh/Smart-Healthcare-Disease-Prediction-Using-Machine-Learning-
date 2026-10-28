"""Disease module registry: feature specifications, validation ranges and data loaders.

Each disease is a self-contained "module" described by a :class:`DiseaseSpec`. The training
pipeline, the inference service, the REST API and the web UI are all driven from this single
registry, so adding a new disease only requires adding one entry here (extensibility goal).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import pandas as pd
from sklearn.datasets import load_breast_cancer

from . import config

# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Feature:
    name: str                      # column name / JSON key
    label: str                     # human readable label for the UI
    kind: str                      # "numeric" | "integer" | "binary" | "category"
    min: float | None = None       # plausible clinical range used for input validation
    max: float | None = None
    unit: str = ""
    help: str = ""
    choices: tuple | None = None   # ((code, label), ...) for binary / category features
    allow_missing: bool = False    # may the user leave it blank (value is then imputed)?

    def to_dict(self) -> dict:
        d = asdict(self)
        d["choices"] = [{"value": v, "label": l} for v, l in self.choices] if self.choices else None
        return d


@dataclass(frozen=True)
class DiseaseSpec:
    key: str
    name: str
    short_description: str
    positive_label: str            # wording for the positive class (risk language, not diagnosis)
    negative_label: str
    features: tuple
    sentinels: dict                # {column: [codes that mean "missing"]}
    source: str
    population_note: str
    loader: Callable[[], tuple]    # () -> (X, y, info)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def feature(self, name: str) -> Feature:
        return next(f for f in self.features if f.name == name)


# --------------------------------------------------------------------------------------
# Loaders. Convention: y == 1 means the condition is PRESENT (positive class).
# --------------------------------------------------------------------------------------

_PIMA_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
              "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


def load_diabetes():
    path = config.DATA_DIR / "pima_diabetes" / "pima-indians-diabetes.data.csv"
    df = pd.read_csv(path, header=None, names=_PIMA_COLS)
    raw_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    y = df.pop("Outcome").astype(int)
    info = {"raw_rows": raw_rows, "duplicates_removed": raw_rows - len(df),
            "label_note": "Outcome=1 -> diabetes within 5 years (as published)."}
    return df, y, info


def load_heart():
    path = config.DATA_DIR / "heart_disease" / "heart.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    raw_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    # DATA DEFECT FOUND DURING PROFILING: in this CSV `target == 1` marks the *healthy* class
    # (those rows have higher max heart rate, lower ST depression, ~4x fewer exercise-induced
    # angina cases). We therefore define disease = 1 - target so that 1 always means "present".
    y = (1 - df.pop("target")).astype(int)
    info = {"raw_rows": raw_rows, "duplicates_removed": raw_rows - len(df),
            "label_note": "Source `target` is inverted in this CSV; disease = 1 - target "
                          "(verified against clinical signals: thalach, oldpeak, exang, ca)."}
    return df, y, info


_BREAST_MEAN10 = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
                  "mean compactness", "mean concavity", "mean concave points",
                  "mean symmetry", "mean fractal dimension"]


def load_breast(feature_set: str = "mean10"):
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    if feature_set == "mean10":
        X = X[_BREAST_MEAN10]
    X.columns = [c.replace(" ", "_") for c in X.columns]
    raw_rows = len(X)
    # sklearn encodes malignant = 0, benign = 1 -> flip so 1 == malignant (condition present)
    y = (1 - bunch.target).astype(int)
    info = {"raw_rows": raw_rows, "duplicates_removed": 0,
            "label_note": "sklearn target is malignant=0/benign=1; flipped so 1 == malignant."}
    return X.reset_index(drop=True), y.reset_index(drop=True), info


# --------------------------------------------------------------------------------------
# Feature specifications
# --------------------------------------------------------------------------------------

DIABETES_FEATURES = (
    Feature("Pregnancies", "Number of pregnancies", "integer", 0, 20),
    Feature("Glucose", "Plasma glucose (2 h OGTT)", "numeric", 40, 300, "mg/dL",
            "Plasma glucose concentration 2 hours into an oral glucose tolerance test.", allow_missing=True),
    Feature("BloodPressure", "Diastolic blood pressure", "numeric", 20, 140, "mmHg", allow_missing=True),
    Feature("SkinThickness", "Triceps skin-fold thickness", "numeric", 5, 100, "mm", allow_missing=True),
    Feature("Insulin", "2-hour serum insulin", "numeric", 10, 900, "µU/mL", allow_missing=True),
    Feature("BMI", "Body mass index", "numeric", 12, 70, "kg/m²", allow_missing=True),
    Feature("DiabetesPedigreeFunction", "Diabetes pedigree function", "numeric", 0.05, 2.5, "",
            "Score summarising diabetes family history."),
    Feature("Age", "Age", "integer", 18, 100, "years"),
)

_YN = ((0, "No"), (1, "Yes"))

HEART_FEATURES = (
    Feature("age", "Age", "integer", 20, 100, "years"),
    Feature("sex", "Sex", "binary", choices=((0, "Female"), (1, "Male"))),
    Feature("cp", "Chest-pain type (dataset code)", "category",
            choices=((0, "Code 0"), (1, "Code 1"), (2, "Code 2"), (3, "Code 3")),
            help="Coded exactly as in the source CSV; the published codebook for this CSV is "
                 "inconsistent with the original UCI coding, so no clinical names are attached."),
    Feature("trestbps", "Resting blood pressure", "numeric", 70, 250, "mmHg"),
    Feature("chol", "Serum cholesterol", "numeric", 100, 600, "mg/dL"),
    Feature("fbs", "Fasting blood sugar > 120 mg/dL", "binary", choices=_YN),
    Feature("restecg", "Resting ECG result (dataset code)", "category",
            choices=((0, "Code 0"), (1, "Code 1"), (2, "Code 2")),
            help="Coded as in the source CSV."),
    Feature("thalach", "Maximum heart rate achieved", "numeric", 60, 220, "bpm"),
    Feature("exang", "Exercise-induced angina", "binary", choices=_YN),
    Feature("oldpeak", "ST depression induced by exercise", "numeric", 0, 7, "mm",
            "ST depression relative to rest."),
    Feature("slope", "Slope of peak-exercise ST segment (dataset code)", "category",
            choices=((0, "Code 0"), (1, "Code 1"), (2, "Code 2")), help="Coded as in the source CSV."),
    Feature("ca", "Major vessels coloured by fluoroscopy", "integer", 0, 3, "",
            "Number of major vessels (0-3).", allow_missing=True),
    Feature("thal", "Thallium stress-test result (dataset code)", "category",
            choices=((1, "Code 1"), (2, "Code 2"), (3, "Code 3")), help="Coded as in the source CSV.",
            allow_missing=True),
)

BREAST_FEATURES = (
    Feature("mean_radius", "Mean radius", "numeric", 5, 35, "", "Mean distance from centre to perimeter."),
    Feature("mean_texture", "Mean texture", "numeric", 5, 50, "", "Std. dev. of grey-scale values."),
    Feature("mean_perimeter", "Mean perimeter", "numeric", 30, 250),
    Feature("mean_area", "Mean area", "numeric", 100, 3000),
    Feature("mean_smoothness", "Mean smoothness", "numeric", 0.03, 0.25, "", "Local variation in radius lengths."),
    Feature("mean_compactness", "Mean compactness", "numeric", 0.005, 0.5, "", "perimeter² / area - 1"),
    Feature("mean_concavity", "Mean concavity", "numeric", 0.0, 0.6, "", "Severity of concave portions."),
    Feature("mean_concave_points", "Mean concave points", "numeric", 0.0, 0.3, "", "Number of concave portions."),
    Feature("mean_symmetry", "Mean symmetry", "numeric", 0.08, 0.4),
    Feature("mean_fractal_dimension", "Mean fractal dimension", "numeric", 0.03, 0.12, "", "Coastline approximation - 1."),
)

# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

DISEASES: dict[str, DiseaseSpec] = {
    "diabetes": DiseaseSpec(
        key="diabetes", name="Diabetes",
        short_description="Risk of diabetes onset from 8 clinical measurements.",
        positive_label="Elevated diabetes risk", negative_label="Lower diabetes risk",
        features=DIABETES_FEATURES,
        sentinels={c: [0] for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]},
        source="Pima Indians Diabetes Database (UCI / NIDDK), 768 patients",
        population_note="Female patients aged 21+ of Pima Indian heritage; may not generalise to other populations.",
        loader=load_diabetes),
    "heart": DiseaseSpec(
        key="heart", name="Heart Disease",
        short_description="Likelihood of coronary heart disease from 13 clinical attributes.",
        positive_label="Elevated heart-disease risk", negative_label="Lower heart-disease risk",
        features=HEART_FEATURES,
        sentinels={"ca": [4], "thal": [0]},
        source="UCI Heart Disease (Cleveland subset), 303 patients",
        population_note="Single-centre (Cleveland Clinic) cohort, mostly older male patients; small sample.",
        loader=load_heart),
    "breast": DiseaseSpec(
        key="breast", name="Breast Cancer",
        short_description="Malignancy risk of a breast mass from 10 cell-nucleus measurements.",
        positive_label="Likely malignant", negative_label="Likely benign",
        features=BREAST_FEATURES,
        sentinels={},
        source="Wisconsin Diagnostic Breast Cancer (WDBC), 569 fine-needle-aspirate samples",
        population_note="Features are computed from digitised FNA images; not applicable to raw patient symptoms.",
        loader=load_breast),
}


def get_spec(key: str) -> DiseaseSpec:
    try:
        return DISEASES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown disease module '{key}'. Available: {sorted(DISEASES)}") from exc

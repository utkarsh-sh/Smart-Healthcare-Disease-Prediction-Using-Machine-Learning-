"""Data-layer tests: loaders, label polarity, sentinel handling and pipeline structure."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.modeling import build_pipeline, model_zoo
from src.registry import DISEASES, get_spec, load_breast
from src.transformers import SentinelToNaN

EXPECTED_ROWS = {"diabetes": 768, "heart": 302, "breast": 569}     # heart: 1 exact duplicate removed


@pytest.mark.parametrize("key", list(DISEASES))
def test_loader_shapes_labels_and_columns(key):
    spec = get_spec(key)
    X, y, info = spec.loader()
    assert len(X) == len(y) == EXPECTED_ROWS[key]
    assert list(X.columns) == spec.feature_names
    assert set(y.unique()) == {0, 1}
    assert not X.isna().any().any(), "raw data should contain no NaN (missing values are coded as sentinels)"
    assert info["raw_rows"] >= len(X)


def test_heart_duplicate_removed_and_reported():
    _, _, info = get_spec("heart").loader()
    assert info["raw_rows"] == 303 and info["duplicates_removed"] == 1


def test_heart_label_polarity_is_clinically_consistent():
    """Regression guard for the inverted `target` column in the source CSV.
    Patients with disease (y=1) must show lower peak heart rate and more exercise angina."""
    X, y, _ = get_spec("heart").loader()
    assert X.loc[y == 1, "thalach"].mean() < X.loc[y == 0, "thalach"].mean()
    assert X.loc[y == 1, "exang"].mean() > X.loc[y == 0, "exang"].mean()
    assert X.loc[y == 1, "oldpeak"].mean() > X.loc[y == 0, "oldpeak"].mean()


def test_breast_positive_class_is_malignant():
    X, y, _ = load_breast("mean10")
    assert int(y.sum()) == 212                       # WDBC has 212 malignant samples
    assert X.loc[y == 1, "mean_radius"].mean() > X.loc[y == 0, "mean_radius"].mean()
    X30, _, _ = load_breast("all30")
    assert X30.shape[1] == 30


def test_pima_diabetes_class_balance():
    _, y, _ = get_spec("diabetes").loader()
    assert (y == 1).sum() == 268 and (y == 0).sum() == 500


def test_sentinel_transformer_replaces_only_listed_columns_and_does_not_mutate_input():
    df = pd.DataFrame({"Glucose": [0, 120], "BMI": [0, 30.0], "Age": [0, 40]})
    original = df.copy()
    out = SentinelToNaN({"Glucose": [0], "BMI": [0]}).fit_transform(df)
    assert out["Glucose"].isna().tolist() == [True, False]
    assert out["BMI"].isna().tolist() == [True, False]
    assert out["Age"].tolist() == [0.0, 40.0]        # 'Age' is not a sentinel column
    pd.testing.assert_frame_equal(df, original)      # input untouched


def test_pima_missing_counts_after_sentinel_conversion():
    spec = get_spec("diabetes")
    X, _, _ = spec.loader()
    missing = SentinelToNaN(spec.sentinels).fit_transform(X).isna().sum()
    assert missing[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].tolist() == [5, 35, 227, 374, 11]
    assert missing["Pregnancies"] == 0               # zero pregnancies is a legitimate value


def test_heart_invalid_codes_become_missing():
    spec = get_spec("heart")
    X, _, _ = spec.loader()
    clean = SentinelToNaN(spec.sentinels).fit_transform(X)
    assert clean["ca"].isna().sum() == int((X["ca"] == 4).sum()) > 0
    assert clean["thal"].isna().sum() == int((X["thal"] == 0).sum()) > 0


@pytest.mark.parametrize("key", list(DISEASES))
def test_pipeline_output_is_finite_and_imputation_is_inside_pipeline(key):
    spec = get_spec(key)
    X, y, _ = spec.loader()
    pipe = build_pipeline(spec.features, spec.sentinels, model_zoo(0)["Logistic Regression"]).fit(X, y)
    transformed = pipe[:-1].transform(X)
    assert np.isfinite(transformed).all()
    # fitting statistics (scaler / imputer / encoder) live in the pipeline => no leakage inside CV folds
    prep = pipe.named_steps["prep"]
    inner = [step for _, t, _ in prep.transformers_ if hasattr(t, "steps") for _, step in t.steps]
    assert any(isinstance(s, (StandardScaler, OneHotEncoder)) for s in inner)
    assert list(pipe.named_steps) == ["sentinel", "prep", "clf"]


@pytest.mark.parametrize("key", list(DISEASES))
def test_registry_specs_are_internally_consistent(key):
    spec = get_spec(key)
    names = spec.feature_names
    assert len(names) == len(set(names))
    for f in spec.features:
        if f.kind in ("numeric", "integer"):
            assert f.min is not None and f.max is not None and f.min < f.max
        else:
            assert f.choices and len({c for c, _ in f.choices}) == len(f.choices)
    assert set(spec.sentinels) <= set(names)


def test_unknown_disease_raises_keyerror():
    with pytest.raises(KeyError):
        get_spec("malaria")


@pytest.mark.parametrize("key", list(DISEASES))
def test_validation_ranges_cover_every_real_dataset_value(key):
    """Data contract: every genuine record must be accepted by the API's input validation
    (sentinel codes excluded - they are entered as blank)."""
    spec = get_spec(key)
    X, _, _ = spec.loader()
    for f in spec.features:
        col = X[f.name]
        col = col[~col.isin(spec.sentinels.get(f.name, []))]
        if f.kind in ("numeric", "integer"):
            assert col.min() >= f.min and col.max() <= f.max, (f.name, col.min(), col.max())
        else:
            assert set(col.unique()) <= {c for c, _ in f.choices}, f.name

# Data codebook and defects found

All datasets are public research data. Source files are in `data/raw/` (breast cancer ships with scikit-learn).

## 1. Pima Indians Diabetes (768 rows, 8 features)
Source: UCI / NIDDK, `pima-indians-diabetes.data.csv` (headerless). Positive class: diabetes within 5 years (268 of 768).

| Feature | Meaning | Unit |
| --- | --- | --- |
| Pregnancies | Number of pregnancies | count |
| Glucose | Plasma glucose, 2 h oral glucose tolerance test | mg/dL |
| BloodPressure | Diastolic blood pressure | mmHg |
| SkinThickness | Triceps skin-fold thickness | mm |
| Insulin | 2-hour serum insulin | µU/mL |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Family-history score | - |
| Age | Age | years |

**Defect:** `0` is used as a missing-value code in Glucose (5 rows), BloodPressure (35), SkinThickness (227), Insulin (374) and BMI (11).
**Handling:** `SentinelToNaN` converts these to missing, then median imputation inside each CV fold. Zero pregnancies is legitimate and left alone.

## 2. UCI Heart Disease, Cleveland subset (303 rows, 1 exact duplicate removed, 13 features)
Source: GitHub copy of the widely used Kaggle/UCI `heart.csv`. After de-duplication 302 rows, 138 positive.

**Defect 1 - inverted label.** In this CSV `target = 1` marks the *healthy* class. Evidence (means by `target`, computed on all 303 rows before the duplicate was removed):

| Signal | target = 0 | target = 1 |
| --- | --- | --- |
| Max heart rate (thalach) | 139.1 | 158.5 |
| ST depression (oldpeak) | 1.59 | 0.58 |
| Exercise-induced angina rate | 55.1% | 13.9% |
| Mean major vessels (ca) | 1.17 | 0.36 |

Clinically, disease lowers peak heart rate and raises ST depression, angina and vessel count, so `target = 0` is the disease class.
**Handling:** `disease = 1 - target`. A regression test (`test_heart_label_polarity_is_clinically_consistent`) guards this.

**Defect 2 - invalid codes:** `ca = 4` (5 rows) and `thal = 0` (2 rows) are not valid values; treated as missing.

**Defect 3 - codebook mismatch:** the categorical codes in this CSV (`cp`, `restecg`, `slope`, `thal`) do not match the original UCI numbering, and the usually quoted codebook is inconsistent with the data.
**Handling:** the models use the codes as one-hot categories (which is safe, since one-hot ignores numeric meaning), and the UI shows "Code 0/1/2/3" instead of inventing clinical names.
**Recommended follow-up:** re-run with the original UCI `processed.cleveland.data` so categories can carry validated clinical names.

## 3. Wisconsin Diagnostic Breast Cancer (569 rows)
Source: `sklearn.datasets.load_breast_cancer`. 212 malignant, 357 benign. scikit-learn encodes malignant = 0; the loader flips it so 1 = malignant.
Used features: the 10 *mean* cell-nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
An ablation with all 30 features is reported in `reports/results.json`.

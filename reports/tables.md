### Datasets

| Module | Raw rows | Duplicates removed | Rows used | Features | Positive / negative | Train / test | Missing after sentinel conversion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetes | 768 | 0 | 768 | 8 | 268 / 500 (34.9%) | 614 / 154 | Glucose: 5, BloodPressure: 35, SkinThickness: 227, Insulin: 374, BMI: 11 |
| Heart Disease | 303 | 1 | 302 | 13 | 138 / 164 (45.7%) | 241 / 61 | ca: 4, thal: 2 |
| Breast Cancer | 569 | 0 | 569 | 10 | 212 / 357 (37.3%) | 455 / 114 | none |

### Baseline CV, default parameters: Diabetes (5-fold x 3 repeats on 614 training rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Majority-class baseline | 65.1% | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic Regression | 78.3% | 0.752 ± 0.077 | 0.573 ± 0.081 | 0.646 ± 0.066 | 0.842 ± 0.027 |
| KNN | 74.2% | 0.653 ± 0.090 | 0.586 ± 0.060 | 0.613 ± 0.054 | 0.798 ± 0.048 |
| SVM (RBF) | 76.3% | 0.712 ± 0.079 | 0.561 ± 0.078 | 0.622 ± 0.052 | 0.832 ± 0.031 |
| Random Forest | 75.6% | 0.680 ± 0.074 | 0.584 ± 0.078 | 0.624 ± 0.059 | 0.823 ± 0.039 |
| Gradient Boosting | 74.9% | 0.664 ± 0.072 | 0.591 ± 0.061 | 0.621 ± 0.044 | 0.820 ± 0.032 |

### Baseline CV, default parameters: Heart Disease (5-fold x 3 repeats on 241 training rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Majority-class baseline | 54.4% | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic Regression | 84.5% | 0.858 ± 0.077 | 0.800 ± 0.064 | 0.825 ± 0.050 | 0.907 ± 0.048 |
| KNN | 84.1% | 0.840 ± 0.070 | 0.812 ± 0.064 | 0.823 ± 0.048 | 0.874 ± 0.045 |
| SVM (RBF) | 83.8% | 0.853 ± 0.087 | 0.794 ± 0.075 | 0.817 ± 0.045 | 0.902 ± 0.043 |
| Random Forest | 82.9% | 0.831 ± 0.097 | 0.800 ± 0.066 | 0.811 ± 0.051 | 0.903 ± 0.042 |
| Gradient Boosting | 80.2% | 0.785 ± 0.045 | 0.785 ± 0.058 | 0.783 ± 0.036 | 0.881 ± 0.044 |

### Baseline CV, default parameters: Breast Cancer (5-fold x 3 repeats on 455 training rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Majority-class baseline | 62.6% | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic Regression | 94.2% | 0.946 ± 0.033 | 0.898 ± 0.038 | 0.920 ± 0.025 | 0.985 ± 0.008 |
| KNN | 93.3% | 0.925 ± 0.041 | 0.894 ± 0.054 | 0.908 ± 0.034 | 0.970 ± 0.021 |
| SVM (RBF) | 94.1% | 0.956 ± 0.036 | 0.882 ± 0.059 | 0.917 ± 0.037 | 0.985 ± 0.010 |
| Random Forest | 94.1% | 0.931 ± 0.040 | 0.912 ± 0.052 | 0.920 ± 0.036 | 0.982 ± 0.013 |
| Gradient Boosting | 94.2% | 0.926 ± 0.036 | 0.920 ± 0.042 | 0.922 ± 0.030 | 0.986 ± 0.010 |

### Grid search: best parameters

| Module | Model | Best parameters | CV ROC-AUC | Combinations | Search time (s) |
| --- | --- | --- | --- | --- | --- |
| Diabetes | Logistic Regression | C=0.1, class_weight=balanced | 0.8450 | 10 | 0.5 |
| Diabetes | KNN | n_neighbors=15, weights=distance | 0.8328 | 12 | 0.6 |
| Diabetes | SVM (RBF) | C=1, gamma=0.01 | 0.8463 | 12 | 1.8 |
| Diabetes | Random Forest | max_depth=4, min_samples_leaf=5, n_estimators=400 | 0.8400 | 18 | 20.3 |
| Diabetes | Gradient Boosting | learning_rate=0.03, max_depth=2, n_estimators=100 | 0.8274 | 8 | 4.2 |
| Heart Disease | Logistic Regression | C=0.1, class_weight=None | 0.9081 | 10 | 0.9 |
| Heart Disease | KNN | n_neighbors=15, weights=distance | 0.8935 | 12 | 0.9 |
| Heart Disease | SVM (RBF) | C=1, gamma=0.01 | 0.9127 | 12 | 1.1 |
| Heart Disease | Random Forest | max_depth=None, min_samples_leaf=5, n_estimators=200 | 0.9170 | 18 | 17.4 |
| Heart Disease | Gradient Boosting | learning_rate=0.03, max_depth=2, n_estimators=200 | 0.8956 | 8 | 3.4 |
| Breast Cancer | Logistic Regression | C=0.1, class_weight=balanced | 0.9852 | 10 | 0.4 |
| Breast Cancer | KNN | n_neighbors=9, weights=uniform | 0.9843 | 12 | 0.4 |
| Breast Cancer | SVM (RBF) | C=100, gamma=0.01 | 0.9872 | 12 | 0.7 |
| Breast Cancer | Random Forest | max_depth=8, min_samples_leaf=1, n_estimators=400 | 0.9846 | 18 | 19.6 |
| Breast Cancer | Gradient Boosting | learning_rate=0.1, max_depth=2, n_estimators=200 | 0.9894 | 8 | 4.2 |

### Selection round on new CV folds: Diabetes

| Candidate | ROC-AUC | F1 | Recall |
| --- | --- | --- | --- |
| Soft-Voting Ensemble (selected) | 0.849 ± 0.032 | 0.655 ± 0.030 | 0.606 ± 0.051 |
| Logistic Regression | 0.845 ± 0.031 | 0.671 ± 0.047 | 0.707 ± 0.059 |
| SVM (RBF) | 0.844 ± 0.029 | 0.624 ± 0.043 | 0.537 ± 0.053 |
| Random Forest | 0.841 ± 0.033 | 0.602 ± 0.044 | 0.522 ± 0.068 |
| Gradient Boosting | 0.840 ± 0.031 | 0.614 ± 0.054 | 0.540 ± 0.079 |
| KNN | 0.836 ± 0.038 | 0.637 ± 0.038 | 0.582 ± 0.045 |

### Selection round on new CV folds: Heart Disease

| Candidate | ROC-AUC | F1 | Recall |
| --- | --- | --- | --- |
| SVM (RBF) (selected) | 0.914 ± 0.048 | 0.830 ± 0.069 | 0.773 ± 0.097 |
| Soft-Voting Ensemble | 0.912 ± 0.044 | 0.824 ± 0.067 | 0.797 ± 0.091 |
| Random Forest | 0.912 ± 0.045 | 0.811 ± 0.062 | 0.797 ± 0.087 |
| Logistic Regression | 0.910 ± 0.051 | 0.834 ± 0.077 | 0.791 ± 0.103 |
| Gradient Boosting | 0.895 ± 0.046 | 0.797 ± 0.062 | 0.776 ± 0.098 |
| KNN | 0.889 ± 0.054 | 0.786 ± 0.063 | 0.742 ± 0.089 |

### Selection round on new CV folds: Breast Cancer

| Candidate | ROC-AUC | F1 | Recall |
| --- | --- | --- | --- |
| Soft-Voting Ensemble | 0.988 ± 0.006 | 0.928 ± 0.028 | 0.916 ± 0.041 |
| SVM (RBF) (selected) | 0.988 ± 0.009 | 0.918 ± 0.027 | 0.898 ± 0.044 |
| Gradient Boosting | 0.987 ± 0.009 | 0.927 ± 0.023 | 0.927 ± 0.037 |
| Logistic Regression | 0.984 ± 0.009 | 0.914 ± 0.031 | 0.902 ± 0.052 |
| Random Forest | 0.982 ± 0.010 | 0.911 ± 0.029 | 0.906 ± 0.037 |
| KNN | 0.981 ± 0.011 | 0.908 ± 0.039 | 0.894 ± 0.052 |

### Hold-out results at the 0.5 decision line (95% bootstrap intervals)

| Module | Selected model | n | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | MCC | Brier | Confusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetes | Soft-Voting Ensemble | 154 | 69.5% (61.7% to 76.6%) | 57.8% (42.5% to 71.1%) | 48.1% (34.4% to 61.4%) | 81.0% | 52.5% (39.2% to 63.5%) | 0.810 (0.739 to 0.872) | 0.306 | 0.170 | TN 81, FP 19, FN 28, TP 26 |
| Heart Disease | SVM (RBF) | 61 | 85.2% (77.0% to 93.4%) | 88.0% (75.0% to 100.0%) | 78.6% (63.0% to 92.3%) | 90.9% | 83.0% (71.1% to 92.3%) | 0.911 (0.835 to 0.970) | 0.704 | 0.116 | TN 30, FP 3, FN 6, TP 22 |
| Breast Cancer | SVM (RBF) | 114 | 97.4% (93.9% to 100.0%) | 97.6% (91.7% to 100.0%) | 95.2% (87.5% to 100.0%) | 98.6% | 96.4% (91.4% to 100.0%) | 0.989 (0.969 to 1.000) | 0.943 | 0.031 | TN 71, FP 1, FN 2, TP 40 |

### Reference only: every candidate on the hold-out (not used for any decision)

| Module | Candidate | Accuracy | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Diabetes | Logistic Regression | 70.8% | 66.7% | 0.615 | 0.810 |
| Diabetes | KNN | 73.4% | 53.7% | 0.586 | 0.807 |
| Diabetes | SVM (RBF) | 70.8% | 50.0% | 0.545 | 0.805 |
| Diabetes | Random Forest | 73.4% | 48.1% | 0.559 | 0.804 |
| Diabetes | Gradient Boosting | 70.8% | 48.1% | 0.536 | 0.810 |
| Diabetes | Soft-Voting Ensemble (selected) | 69.5% | 48.1% | 0.525 | 0.810 |
| Heart Disease | Logistic Regression | 85.2% | 78.6% | 0.830 | 0.922 |
| Heart Disease | KNN | 78.7% | 71.4% | 0.755 | 0.907 |
| Heart Disease | SVM (RBF) (selected) | 85.2% | 78.6% | 0.830 | 0.911 |
| Heart Disease | Random Forest | 78.7% | 71.4% | 0.755 | 0.889 |
| Heart Disease | Gradient Boosting | 80.3% | 71.4% | 0.769 | 0.919 |
| Heart Disease | Soft-Voting Ensemble | 80.3% | 71.4% | 0.769 | 0.918 |
| Breast Cancer | Logistic Regression | 93.9% | 95.2% | 0.920 | 0.985 |
| Breast Cancer | KNN | 95.6% | 95.2% | 0.941 | 0.985 |
| Breast Cancer | SVM (RBF) (selected) | 97.4% | 95.2% | 0.964 | 0.989 |
| Breast Cancer | Random Forest | 93.9% | 90.5% | 0.916 | 0.987 |
| Breast Cancer | Gradient Boosting | 95.6% | 95.2% | 0.941 | 0.983 |
| Breast Cancer | Soft-Voting Ensemble | 96.5% | 95.2% | 0.952 | 0.990 |

### Screening line versus the 0.5 decision line (hold-out)

| Module | Screening threshold | Out-of-fold recall (train) | Hold-out recall | Hold-out specificity | Hold-out precision | Share flagged | Recall at 0.5 | Specificity at 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetes | 0.262 | 90.2% | 88.9% | 60.0% | 54.5% | 57.1% | 48.1% | 81.0% |
| Heart Disease | 0.292 | 90.0% | 89.3% | 69.7% | 71.4% | 57.4% | 78.6% | 90.9% |
| Breast Cancer | 0.519 | 90.0% | 95.2% | 98.6% | 97.6% | 36.0% | 95.2% | 98.6% |

### Stability over 10 random 80/20 splits: mean ± SD (min to max)

| Module | Splits | Accuracy | Recall | F1 | ROC-AUC | Seed-42 accuracy (reported split) |
| --- | --- | --- | --- | --- | --- | --- |
| Diabetes | 10 | 75.8% ± 2.9% (71.4% to 80.5%) | 58.1% ± 6.7% (50.0% to 68.5%) | 62.7% ± 4.3% (57.4% to 70.6%) | 0.835 ± 0.031 (0.782 to 0.880) | 69.5% |
| Heart Disease | 10 | 85.9% ± 3.2% (82.0% to 91.8%) | 81.1% ± 5.8% (71.4% to 89.3%) | 84.0% ± 3.7% (79.2% to 90.9%) | 0.921 ± 0.020 (0.883 to 0.940) | 85.2% |
| Breast Cancer | 10 | 94.6% ± 1.2% (92.1% to 96.5%) | 90.2% ± 3.5% (85.7% to 95.2%) | 92.6% ± 1.7% (88.9% to 95.2%) | 0.991 ± 0.005 (0.982 to 0.999) | 97.4% |

### Preprocessing ablations (5-fold x 3 CV on the training split)

| Module | Variant | Model | ROC-AUC | F1 | Recall |
| --- | --- | --- | --- | --- | --- |
| Diabetes | zeros_treated_as_missing (ours) | Logistic Regression | 0.842 ± 0.027 | 0.646 ± 0.066 | 0.573 ± 0.081 |
| Diabetes | zeros_treated_as_missing (ours) | Random Forest | 0.823 ± 0.039 | 0.624 ± 0.059 | 0.584 ± 0.078 |
| Diabetes | zeros_kept_as_values | Logistic Regression | 0.831 ± 0.041 | 0.648 ± 0.057 | 0.578 ± 0.064 |
| Diabetes | zeros_kept_as_values | Random Forest | 0.819 ± 0.044 | 0.617 ± 0.049 | 0.564 ± 0.066 |
| Heart Disease | invalid_codes_ca4_thal0_to_missing (ours) | Logistic Regression | 0.907 ± 0.048 | 0.825 ± 0.050 | 0.800 ± 0.064 |
| Heart Disease | invalid_codes_ca4_thal0_to_missing (ours) | Random Forest | 0.903 ± 0.042 | 0.811 ± 0.051 | 0.800 ± 0.066 |
| Heart Disease | invalid_codes_kept_as_values | Logistic Regression | 0.907 ± 0.048 | 0.828 ± 0.046 | 0.806 ± 0.058 |
| Heart Disease | invalid_codes_kept_as_values | Random Forest | 0.906 ± 0.043 | 0.808 ± 0.056 | 0.797 ± 0.075 |
| Breast Cancer | mean10_features (ours) | Logistic Regression | 0.985 ± 0.008 | 0.920 ± 0.025 | 0.898 ± 0.038 |
| Breast Cancer | mean10_features (ours) | Random Forest | 0.982 ± 0.013 | 0.920 ± 0.036 | 0.912 ± 0.052 |
| Breast Cancer | all30_features | Logistic Regression | 0.996 ± 0.005 | 0.967 ± 0.020 | 0.957 ± 0.031 |
| Breast Cancer | all30_features | Random Forest | 0.990 ± 0.008 | 0.939 ± 0.032 | 0.925 ± 0.053 |

### Permutation importance, top 5: Diabetes

| Rank | Feature | Drop in ROC-AUC when shuffled |
| --- | --- | --- |
| 1 | Plasma glucose (2 h OGTT) | 0.124 ± 0.028 |
| 2 | Body mass index | 0.030 ± 0.016 |
| 3 | Number of pregnancies | 0.014 ± 0.007 |
| 4 | Age | 0.010 ± 0.005 |
| 5 | Diabetes pedigree function | 0.006 ± 0.006 |

### Permutation importance, top 5: Heart Disease

| Rank | Feature | Drop in ROC-AUC when shuffled |
| --- | --- | --- |
| 1 | Major vessels coloured by fluoroscopy | 0.083 ± 0.029 |
| 2 | Chest-pain type (dataset code) | 0.031 ± 0.013 |
| 3 | ST depression induced by exercise | 0.013 ± 0.010 |
| 4 | Maximum heart rate achieved | 0.012 ± 0.014 |
| 5 | Thallium stress-test result (dataset code) | 0.007 ± 0.018 |

### Permutation importance, top 5: Breast Cancer

| Rank | Feature | Drop in ROC-AUC when shuffled |
| --- | --- | --- |
| 1 | Mean concavity | 0.040 ± 0.011 |
| 2 | Mean texture | 0.035 ± 0.010 |
| 3 | Mean area | 0.031 ± 0.012 |
| 4 | Mean smoothness | 0.016 ± 0.008 |
| 5 | Mean concave points | 0.014 ± 0.006 |

### Model-level cost (in-process, one row per call)

| Module | Selected model | Final fit (s) | Whole protocol (s) | Model file (KB) | Predict p50 (ms) | p95 (ms) | p99 (ms) | Batch throughput (rows/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetes | Soft-Voting Ensemble | 0.72 | 104 | 289 | 35.95 | 50.63 | 54.44 | 3500 |
| Heart Disease | SVM (RBF) | 0.03 | 65 | 7 | 6.27 | 9.86 | 10.41 | 5592 |
| Breast Cancer | SVM (RBF) | 0.02 | 70 | 7 | 2.35 | 3.90 | 6.88 | 46506 |

### HTTP latency, one request at a time (includes TCP connect and SQLite write)

| Module | Mode | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) | HTTP errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetes | with explanation | 300 | 84.6 | 108.4 | 111.9 | 114.6 | 0 |
| Diabetes | without explanation | 300 | 41.8 | 56.3 | 64.1 | 74.3 | 0 |
| Heart Disease | with explanation | 300 | 17.7 | 23.0 | 26.4 | 28.3 | 0 |
| Heart Disease | without explanation | 300 | 10.2 | 13.2 | 14.9 | 18.0 | 0 |
| Breast Cancer | with explanation | 300 | 10.2 | 12.8 | 15.2 | 21.9 | 0 |
| Breast Cancer | without explanation | 300 | 5.9 | 7.6 | 9.7 | 12.5 | 0 |

### Concurrent load, mixed modules, explanation off

| Client threads | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | HTTP errors | Server CPU seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 400 | 47.9 | 10.7 | 53.9 | 59.7 | 0 | 7.7 |
| 4 | 400 | 37.1 | 46.3 | 289.9 | 329.1 | 0 | 11.7 |
| 8 | 400 | 36.1 | 66.4 | 635.7 | 722.2 | 0 | 11.8 |

### Batch endpoint

| Module | Rows per request | Total (ms) | Rows/s | Status |
| --- | --- | --- | --- | --- |
| Diabetes | 100 | 44 | 2260 | 200 |
| Diabetes | 500 | 58 | 8666 | 200 |
| Heart Disease | 100 | 12 | 8675 | 200 |
| Heart Disease | 500 | 22 | 22959 | 200 |
| Breast Cancer | 100 | 7 | 14941 | 200 |
| Breast Cancer | 500 | 16 | 31728 | 200 |

Cold start (process launch to healthy): 1.61 s; model load: 0.21 s; server memory idle 168 MB, peak 177 MB; rows written to SQLite during the benchmark: 3060.

Environment: Intel(R) Xeon(R) Processor @ 2.10GHz, 2 logical cores, 7.8 GB RAM, Linux-6.18.44-fc-v180-x86_64-with-glibc2.39, Python 3.13.18, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5. Total training wall time 239 s; training-process peak RSS 184 MB.

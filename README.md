# DIAA—Autism screening classifier (broad overview)

This folder contains a **Jupyter** workflow for the project: it builds a **supervised** model that predicts whether a row is labeled as ASD (autistic spectrum) or not, using **questionnaire item scores (A1–A10)** and **metadata**, then reports how well it matches the labels and **which inputs the model leans on**.

> **This is a machine-learning study on a public tabular dataset, not a clinical validation.** High scores in cross-validation do not mean the model is fit for real screening without further external validation, fairness work, and domain review.

## What is in the repo

| Item | Description |
|------|-------------|
| `Autism_Clasifier_Model.ipynb` | Main notebook: data prep, model, cross-validation, permutation importance, optional SHAP, supporting plots. |
| `Documentación Proyecto DIIA/` | LaTeX / academic write-up (separate from the notebook). |

## Data

- **Source:** [UCI “Autism Screening Adult”](https://archive.ics.uci.edu/dataset/426/autism+screening+adult) (ID **426**), loaded in code with **`ucimlrepo`** (`fetch_ucirepo(id=426)`).
- **Target:** `class_binary` (1 = YES / ASD in this dataset’s coding, 0 = NO).
- **Rows:** After cleaning, the modeling step uses on the order of **~600** complete cases (e.g. **n = 608** in a recent run, with about **30%** prevalence of the positive class—exact numbers can change if you re-run with different cleaning).
- **Features:** A1_Score–A10_Score, plus fields such as **age**, **gender**, **ethnicity**, **country_of_res**, **jaundice**, **family_pdd**, **used_app_before**, **age_desc**, **relation**.
- **Excluded from predictors:** `result` and the original label column(s) are not fed as inputs to the classifier, so the model is not given a precomputed summary that duplicates the same items; see the markdown in the notebook for the rationale.

 Cleaning includes dropping rows with missing values in key fields and handling placeholders like `?` / `NA` when applicable.

## Modeling approach (high level)

1. **Preprocessing:** A **scikit-learn** `Pipeline` with a `ColumnTransformer`: numeric features are **standardized**; categorical columns are **one-hot encoded** (with a `sparse` / `sparse_output` compatibility note in code for different sklearn versions).

2. **Classifier:** **Random forest** (many trees, depth and leaf size capped, `class_weight="balanced"` to reduce majority-class bias, parallel `n_jobs` where set).

3. **Evaluation:** **Stratified k-fold** cross-validation (e.g. **5** folds) so each split keeps similar class balance. Reported metrics typically include **AUROC**, **F1**, **precision**, and **recall** (mean ± std across folds, out-of-fold).

4. **Interpretation:** **Permutation importance** (AUROC drop when a column is shuffled on held-out data), **averaged across folds**; a bar plot for A-questions vs metadata. **Spearman** correlation of A1–A10 importance *ranks* across fold pairs checks whether the item ordering is **stable**. **SHAP** is **optional** if the `shap` package is installed.

## How to run

1. Use **Python 3** with a scientific stack: **pandas**, **numpy**, **scikit-learn**, **matplotlib** (and **statsmodels** / other libs if earlier cells in the notebook import them). Install **`ucimlrepo`** for the dataset fetch.
2. Open `Autism_Clasifier_Model.ipynb` in **Jupyter**, **JupyterLab**, or **VS Code**, then run cells from the top. Re-running the main modeling cell updates printed metrics and figures; numbers may differ slightly with library versions or data refresh.

## Limitations (keep in mind in reports and slides)

- **Overfitting to this table:** Strong in-sample / CV performance does not by itself generalize to new countries, sites, or populations.
- **Imbalanced labels and metrics:** Plain accuracy is misleading; the notebook emphasizes AUROC, F1, precision, and recall.
- **Permutation / SHAP:** They describe **model reliance** in this setup, not causal effects of any question on autism.
- **No substitute for a held-out or prospective test set** is implied if you need a “final” deployment claim; cross-validation is for **estimation**, not a locked-down validation protocol unless you add one.

## Citation (dataset)

If you cite the dataset, use the UCI page and DOI given there, e.g. **DOI: [10.24432/C5F019](https://doi.org/10.24432/C5F019)** and creator **Fadi Thabtah**, as shown in the UCI entry for [Autism Screening Adult](https://archive.ics.uci.edu/dataset/426/autism+screening+adult).

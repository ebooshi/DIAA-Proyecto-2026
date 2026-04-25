import warnings, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings("ignore")

# --- Load and clean data ---
autism_screening_adult = fetch_ucirepo(id=426)
df = pd.concat([autism_screening_adult.data.features, autism_screening_adult.data.targets], axis=1)
df = df.replace({"?": None, "NA": None, "N/A": None, "Unknown": None}).dropna()
df = df[df['age'] != 383.0]
df['class_binary']    = df['class'].map({'YES': 1, 'NO': 0}).astype(int)
df['jaundice']        = df['jaundice'].map({'yes': 1, 'no': 0}).astype(int)
df['family_pdd']      = df['family_pdd'].map({'yes': 1, 'no': 0}).astype(int)
df['used_app_before'] = df['used_app_before'].map({'yes': 1, 'no': 0}).astype(int)

# --- Features ---
ITEM_COLS   = [f"A{i}_Score" for i in range(1, 11)]
NUMERIC     = ITEM_COLS + ["age", "jaundice", "family_pdd", "used_app_before"]
CATEGORICAL = ["gender", "ethnicity", "country_of_res", "age_desc", "relation"]
X = df[[c for c in NUMERIC + CATEGORICAL if c in df.columns]].copy()
y = df["class_binary"].astype(int)

# --- Pipeline ---
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

pre = ColumnTransformer([
    ("num", StandardScaler(), [c for c in NUMERIC if c in X.columns]),
    ("cat", ohe,              [c for c in CATEGORICAL if c in X.columns]),
], verbose_feature_names_out=False)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
pipe = Pipeline([("pre", pre), ("clf", clf)])

# --- Permutation importance over 5 folds ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
importances = []
for train_idx, test_idx in skf.split(X, y):
    m = clone(pipe)
    m.fit(X.iloc[train_idx], y.iloc[train_idx])
    pi = permutation_importance(m, X.iloc[test_idx], y.iloc[test_idx],
                                n_repeats=15, random_state=42,
                                n_jobs=-1, scoring="roc_auc")
    importances.append(pi.importances_mean)

imp = np.vstack(importances)
imp_df = (pd.DataFrame({"feature": list(X.columns),
                         "importance_mean": imp.mean(0),
                         "importance_std":  imp.std(0)})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))

# --- Plot and save ---
fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(imp_df))))
ax.barh(imp_df["feature"], imp_df["importance_mean"],
        xerr=imp_df["importance_std"], capsize=2, color="steelblue")
ax.invert_yaxis()
ax.set_xlabel("Drop in AUROC when feature is shuffled (mean ± std across folds)")
ax.set_title("Permutation importance — AQ items and metadata")
fig.tight_layout()

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagens")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rf_permutation_importance.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("Saved:", out_path)
plt.show()

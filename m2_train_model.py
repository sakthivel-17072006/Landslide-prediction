"""MILESTONE 2 - ML model predicts landslide risk.  Run: python m2_train_model.py"""
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, recall_score
from common import *
try:
    from xgboost import XGBClassifier as GB; GB_NAME = "XGBoost"
except ImportError:
    GB, GB_NAME = GradientBoostingClassifier, "GradientBoosting"
if not (ROOT / "data/landslide_dataset.csv").exists(): raise SystemExit("Dataset missing - run python m1_real_data.py first.")
df = pd.read_csv(ROOT / "data/landslide_dataset.csv")
Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df.landslide, test_size=.2, stratify=df.landslide, random_state=1)
models = {"LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
          "RandomForest": RandomForestClassifier(300, class_weight="balanced", random_state=1), GB_NAME: GB(random_state=1)}
out, best, best_auc = [], None, 0
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; auc = roc_auc_score(yte, p)
    out.append(f"{name}: ROC-AUC={auc:.3f} recall@0.4={recall_score(yte, p > .4):.3f}")
    if auc > best_auc: best, best_name, best_auc = m, name, auc
out.append(f"\nBest: {best_name}\n" + classification_report(yte, best.predict(Xte)))
joblib.dump(best, ROOT / "models/landslide_model.joblib")
(ROOT / "models/best_name.txt").write_text(best_name)
(ROOT / "output/metrics.txt").write_text("\n".join(out)); print("\n".join(out))

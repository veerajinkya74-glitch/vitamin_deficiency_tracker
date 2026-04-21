import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("vitamin_dataset.csv")
X  = df.drop("disease_diagnosis", axis=1)
y  = df["disease_diagnosis"]

print("=" * 55)
print("  Vitamin AI — Model Training")
print("=" * 55)
print(f"  Dataset : {df.shape[0]} rows, {df.shape[1]-1} features")
print(f"\n  Class distribution:")
for cls, count in y.value_counts().items():
    print(f"    {cls:<32} {count} rows")
print()

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ── Model definitions ──────────────────────────────────────────────────────────
models = {
    "rf": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42
    ),
    "nb": BernoulliNB(
        alpha=1.0
    ),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=1.0,
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ])
}

# ── Cross-validation ───────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Train, evaluate, save ──────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    gap       = train_acc - test_acc

    print(f"{'─' * 55}")
    print(f"  {name.upper()} Model")
    print(f"{'─' * 55}")
    print(f"  Train accuracy : {train_acc:.2%}")
    print(f"  Test  accuracy : {test_acc:.2%}")
    print(f"  CV    accuracy : {cv_scores.mean():.2%}  ±  {cv_scores.std():.2%}")
    print(f"  Overfit gap    : {gap*100:.1f}pp  {'⚠  overfit' if gap > 0.10 else '✅ ok'}")
    print()
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

    with open(f"models/{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"  ✅ Saved → models/{name}_model.pkl\n")

print("=" * 55)
print("  All models saved.")
print("  Report the CV accuracy — it is the most reliable number.")
print("=" * 55)
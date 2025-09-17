import os
import joblib
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ========================
# Train + Evaluate Function
# ========================
def train_and_evaluate(X, y, dataset_name, model_path, schema_path):
    print(f"\n🔹 Training on {dataset_name} dataset...\n")

    print("📊 Class balance before resampling:")
    print(y.value_counts())

    # Apply SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("\n📊 Class balance after SMOTE:")
    print(y_res.value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
    )

    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print(f"RandomForest AUC ({dataset_name}): {rf_auc:.4f}")
    print("\n🔎 Classification Report (RandomForest):")
    print(classification_report(y_test, rf_preds))
    print("\n🧾 Confusion Matrix (RandomForest):")
    print(confusion_matrix(y_test, rf_preds))

    # XGBoost
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

    print(f"\nXGBoost AUC ({dataset_name}): {xgb_auc:.4f}")
    print("\n🔎 Classification Report (XGBoost):")
    print(classification_report(y_test, xgb_preds))
    print("\n🧾 Confusion Matrix (XGBoost):")
    print(confusion_matrix(y_test, xgb_preds))

    # Pick best model
    if rf_auc >= xgb_auc:
        best_model = rf
        best_auc = rf_auc
        model_name = "RandomForest"
    else:
        best_model = xgb
        best_auc = xgb_auc
        model_name = "XGBClassifier"

    print(f"✅ Using {model_name} for {dataset_name} (AUC={best_auc:.4f})")

    # Save model bundle
    model_bundle = {
        "model": best_model,
        "schema": {"columns": list(X.columns)},
    }
    joblib.dump(model_bundle, model_path)

    # Save schema as JSON (for backend schema checks)
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    with open(schema_path, "w") as f:
        json.dump({"columns": list(X.columns)}, f, indent=4)

    print(f"💾 Model saved at {model_path}")
    print(f"📜 Schema saved at {schema_path}")

    return model_bundle


# ========================
# Preprocessing
# ========================
def preprocess_data(df, target_col):
    # Drop transaction ID if present
    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert datetime columns
    for col in X.columns:
        if "Date" in col or "date" in col:
            print("⏳ Extracting datetime features from", col)
            X[col] = pd.to_datetime(X[col], errors="coerce")
            X[col + "_year"] = X[col].dt.year
            X[col + "_month"] = X[col].dt.month
            X[col + "_day"] = X[col].dt.day
            X[col + "_hour"] = X[col].dt.hour
            X = X.drop(columns=[col])

    # Encode categorical string columns
    for col in X.select_dtypes(include=["object"]).columns:
        print(f"🔄 Encoding categorical column: {col}")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y


# ========================
# Main
# ========================
def load_and_train():
    data_files = {
        "creditcard": {
            "file": "/home/ranchitarora/code/ccfd/backend/creditcard.csv",
            "target": "Class",
        },
        "merchant": {
            "file": "/home/ranchitarora/code/ccfd/backend/creditcard_2023.csv",
            "target": "IsFraud",
        },
    }

    models = {}
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/schemas", exist_ok=True)

    for name, info in data_files.items():
        if not os.path.exists(info["file"]):
            print(f"❌ {info['file']} not found, skipping...")
            continue

        df = pd.read_csv(info["file"])
        if info["target"] not in df.columns:
            print(f"❌ '{info['target']}' column not found in {info['file']}")
            continue

        X, y = preprocess_data(df, info["target"])

        model_path = os.path.join("models", f"{name}_model.pkl")
        schema_path = os.path.join("models/schemas", f"{name}_schema.json")

        models[name] = train_and_evaluate(X, y, name, model_path, schema_path)

    # Save combined bundle (for convenience)
    bundle_path = os.path.join("models", "creditcard_model.pkl")
    joblib.dump(models, bundle_path)
    print(f"💾 All models bundled into {bundle_path}")


if __name__ == "__main__":
    load_and_train()

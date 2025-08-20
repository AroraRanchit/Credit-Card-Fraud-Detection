from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os
from trainer import FraudDetectionTrainer  # make sure FraudDetectionTrainer is imported

app = FastAPI()

MODEL_PATH = "fraud_detection_model.pkl"


def load_model_bundle():
    """Load model bundle (model + schema)."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def check_schema_compatibility(df, saved_schema):
    """Check if uploaded dataset schema matches saved schema."""
    new_schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    return new_schema == saved_schema


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load uploaded dataset
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
    else:
        df = pd.read_excel(file.file)

    if "Class" not in df.columns:
        return {"error": "❌ No 'Class' column found in uploaded file"}

    bundle = load_model_bundle()

    # If model exists, check schema
    if bundle:
        model, schema = bundle["model"], bundle["schema"]

        if not check_schema_compatibility(df.drop("Class", axis=1), schema):
            # Schema mismatch → retrain
            print("⚠️ Schema mismatch detected → retraining model...")
            tmp_path = "uploads/latest_training_file.csv"
            df.to_csv(tmp_path, index=False)

            trainer = FraudDetectionTrainer(tmp_path)
            trainer.train(n_trials=50)  # faster retrain with fewer trials

            bundle = load_model_bundle()
            model = bundle["model"]

    else:
        # No saved model yet → train
        print("⚠️ No saved model found → training new model...")
        tmp_path = "uploads/first_training_file.csv"
        df.to_csv(tmp_path, index=False)

        trainer = FraudDetectionTrainer(tmp_path)
        trainer.train(n_trials=50)

        bundle = load_model_bundle()
        model = bundle["model"]

    # Prepare data for prediction
    X = df.drop("Class", axis=1)
    y_true = df["Class"]

    y_pred = model.predict(X)

    acc = (y_pred == y_true).mean()

    return {
        "accuracy": round(acc, 4),
        "predictions": y_pred.tolist(),
        "rows": len(df)
    }

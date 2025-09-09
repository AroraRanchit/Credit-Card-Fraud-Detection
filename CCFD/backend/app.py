# backend/app.py

from typing import Optional, Dict, Any, Tuple, List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from io import BytesIO

# Charts (for email analytics)
import io as pyio
import base64
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv

# Email (optional)
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr

# =========================
# Paths & constants
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SCHEMA_FALLBACK_DIR = os.path.join(MODEL_DIR, "schemas")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTED_DATASETS = ["creditcard", "merchant"]

# Columns we always ignore if present
DROP_ALWAYS = ["TransactionType", "id", "Location"]

# =========================
# Load env
# =========================
load_dotenv(os.path.join(BASE_DIR, ".env"))

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
MAIL_STARTTLS = os.getenv("MAIL_STARTTLS", "True").lower() == "true"
MAIL_SSL_TLS = os.getenv("MAIL_SSL_TLS", "False").lower() == "true"

EMAIL_ENABLED = all([MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM, MAIL_SERVER])

fm: Optional[FastMail] = None
if EMAIL_ENABLED:
    conf = ConnectionConfig(
        MAIL_USERNAME=MAIL_USERNAME,
        MAIL_PASSWORD=MAIL_PASSWORD,
        MAIL_FROM=MAIL_FROM,
        MAIL_SERVER=MAIL_SERVER,
        MAIL_PORT=MAIL_PORT,
        MAIL_STARTTLS=MAIL_STARTTLS,
        MAIL_SSL_TLS=MAIL_SSL_TLS,
        USE_CREDENTIALS=True,
    )
    fm = FastMail(conf)

# =========================
# App & CORS
# =========================
app = FastAPI(title="Credit Card Fraud Detection API 🚀")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Middleware: log raw /predict requests
# =========================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path == "/predict":
        body = await request.body()
        try:
            print("RAW REQUEST BODY:", body.decode("utf-8")[:2000])  # cap to avoid huge logs
        except Exception:
            print("RAW REQUEST BODY: <binary>")
        print("HEADERS:", dict(request.headers))
    response = await call_next(request)
    return response

# =========================
# Utilities
# =========================
def _read_any_table(upload: UploadFile) -> pd.DataFrame:
    """Read CSV/XLS/XLSX into DataFrame."""
    raw = upload.file.read()
    name = (upload.filename or "").lower()
    upload.file.seek(0)
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(raw))
    if name.endswith(".xls") or name.endswith(".xlsx"):
        return pd.read_excel(BytesIO(raw))
    raise ValueError("Unsupported file format (CSV, XLS, XLSX only).")


def _load_sidecar_schema(dataset: str) -> Optional[Dict[str, Any]]:
    schema_json = os.path.join(SCHEMA_FALLBACK_DIR, f"{dataset}_schema.json")
    if os.path.exists(schema_json):
        with open(schema_json, "r") as f:
            schema = json.load(f)
        if "columns" in schema and isinstance(schema["columns"], list):
            return schema
    return None


def _normalize_bundle(obj: Any, dataset: str) -> Dict[str, Any]:
    """Wrap model with schema if needed."""
    if isinstance(obj, dict) and "model" in obj:
        schema = obj.get("schema", {})
        cols = schema.get("columns") if isinstance(schema, dict) else None
        if not cols:
            raise ValueError(f"Model bundle for '{dataset}' missing schema['columns'].")
        return {"model": obj["model"], "schema": {"columns": list(cols)}}

    # If raw model, try sidecar schema
    sidecar = _load_sidecar_schema(dataset)
    if sidecar is None:
        raise ValueError(
            f"Loaded raw model for '{dataset}' but no schema found. "
            f"Provide models/schemas/{dataset}_schema.json or retrain with schema."
        )
    return {"model": obj, "schema": sidecar}


def _ingest_path(path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load model(s) from joblib path."""
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)

    # Multi-dataset bundle
    if isinstance(obj, dict) and any(k in SUPPORTED_DATASETS for k in obj.keys()):
        out = {}
        for ds, inner in obj.items():
            if ds in SUPPORTED_DATASETS:
                out[ds] = _normalize_bundle(inner, ds)
        return out

    # Single dataset model
    dataset = os.path.splitext(os.path.basename(path))[0].replace("_model", "")
    if dataset in SUPPORTED_DATASETS:
        return {dataset: _normalize_bundle(obj, dataset)}
    return None


def _detect_dataset(
    df: pd.DataFrame, bundles: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Detect dataset based on schema overlap."""
    exact, scored = [], []
    for ds, bundle in bundles.items():
        expected = bundle["schema"]["columns"]
        if all(c in df.columns for c in expected):
            exact.append((ds, expected))
        else:
            inter = len([c for c in expected if c in df.columns])
            scored.append((inter, ds, expected))

    if exact:
        exact.sort(key=lambda x: len(x[1]), reverse=True)
        return exact[0][0], exact[0][1]

    if scored:
        scored.sort(reverse=True)
        top_intersection, ds, expected = scored[0]
        if top_intersection >= 2:
            return ds, expected
    return None, None

def _align_columns_flexible(df: pd.DataFrame, expected_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Align uploaded data to expected columns (drop DROP_ALWAYS, fill missing with 0, ignore extras)."""
    # Drop noisy columns if present
    for col in DROP_ALWAYS:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = pd.DataFrame(index=df.index)
    missing = []
    for col in expected_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0
            missing.append(col)

    # --- NEW: Coerce dtypes ---
    for col in X.columns:
        # If column looks numeric, force numeric, else string
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        else:
            X[col] = X[col].astype(str)

    return X, missing


def _predict_probabilities(model: Any, X: pd.DataFrame) -> pd.Series:
    """Compute fraud probabilities."""
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        m, M = raw.min(), raw.max()
        denom = (M - m) if (M - m) > 1e-12 else 1.0
        return pd.Series((raw - m) / denom, index=X.index)
    return pd.Series(model.predict(X).astype(float), index=X.index)

# -------- Email helpers with analytics --------
def _generate_fraud_analytics_charts(df: pd.DataFrame) -> str:
    """Generate analytics charts and return as inline HTML."""
    # Fraud vs Legit
    fraud_counts = df["prediction"].value_counts().reindex([0, 1]).fillna(0)
    plt.figure(figsize=(4, 4))
    plt.pie(
        fraud_counts,
        labels=["Legit", "Fraud"],
        autopct="%1.1f%%",
    )
    plt.title("Fraud vs Legit Transactions")
    buf1 = pyio.BytesIO()
    plt.savefig(buf1, format="png", bbox_inches="tight")
    buf1.seek(0)
    pie_b64 = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close()

    # Amount distribution (if present)
    amount_b64 = None
    if "Amount" in df.columns:
        plt.figure(figsize=(6, 4))
        try:
            sns.histplot(
                data=df, x="Amount", hue="prediction", bins=30, kde=True, alpha=0.6
            )
        except Exception:
            plt.hist(df["Amount"].values, bins=30)
        plt.title("Transaction Amount Distribution (Fraud vs Legit)")
        buf2 = pyio.BytesIO()
        plt.savefig(buf2, format="png", bbox_inches="tight")
        buf2.seek(0)
        amount_b64 = base64.b64encode(buf2.read()).decode("utf-8")
        plt.close()

    html = f"""
    <h2>📊 Credit Card Fraud Analytics Report</h2>
    <h3>Fraud vs Legit Distribution</h3>
    <img src="data:image/png;base64,{pie_b64}" style="width:300px;">
    {("<h3>Transaction Amount Distribution</h3><img src='data:image/png;base64," + amount_b64 + "' style='width:500px;'>") if amount_b64 else ""}
    <br><p>✅ Report generated automatically by Fraud Detection API.</p>
    """
    return html


async def _send_fraud_report(email: EmailStr, filename: str, df: pd.DataFrame, dataset_name: str, name: str):
    """Send fraud analytics report with charts via email (if email is enabled)."""
    if not (EMAIL_ENABLED and fm and email):
        return
    total = len(df)
    fraud_count = int((df["prediction"] == 1).sum())
    fraud_percent = round((fraud_count / total) * 100, 2) if total else 0.0

    analytics_html = _generate_fraud_analytics_charts(df)
    summary_html = f"""
    <h3>Summary</h3>
    <p><b>User:</b> {name}</p>
    <p><b>Dataset:</b> {dataset_name}</p>
    <p><b>File analyzed:</b> {filename}</p>
    <p><b>Total transactions:</b> {total}</p>
    <p><b>Fraudulent transactions detected:</b> {fraud_count} ({fraud_percent}%)</p>
    """
    final_html = summary_html + analytics_html

    message = MessageSchema(
        subject="Your Credit Card Fraud Analysis - Report",
        recipients=[email],
        body=final_html,
        subtype="html"
    )
    try:
        await fm.send_message(message)
    except Exception as e:
        print(f"⚠️ Email sending failed: {e}")

# =========================
# Load Models
# =========================
MODELS: Dict[str, Dict[str, Any]] = {}

# Load per-dataset models
for ds in SUPPORTED_DATASETS:
    ingested = _ingest_path(os.path.join(MODEL_DIR, f"{ds}_model.pkl"))
    if ingested:
        MODELS.update(ingested)

# Load combined bundle (optional legacy name)
bundle_path = os.path.join(MODEL_DIR, "fraud_detection_model.pkl")
ingested_bundle = _ingest_path(bundle_path)
if ingested_bundle:
    for k, v in ingested_bundle.items():
        MODELS.setdefault(k, v)

if not MODELS:
    raise FileNotFoundError(
        "⚠️ No models found.\n"
        "Expected one of:\n"
        " - models/creditcard_model.pkl\n"
        " - models/merchant_model.pkl\n"
        " - models/fraud_detection_model.pkl (dict with both)\n"
        "Each must be {'model','schema':{'columns':[...]}} or raw estimator with sidecar schema."
    )

# =========================
# Routes
# =========================
@app.get("/")
async def root():
    return {"ok": True, "message": "✅ API is running", "models_loaded": list(MODELS.keys())}


@app.get("/health")
async def health():
    return {"ok": True, "models_loaded": list(MODELS.keys())}


@app.get("/schemas")
async def schemas():
    return {ds: {"expected_columns": bundle["schema"]["columns"]} for ds, bundle in MODELS.items()}


@app.post("/predict")
async def predict(
    name: str = Form(...),
    email: str = Form(...),
    consent: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Upload CSV/XLS/XLSX -> detect dataset -> predict fraud probability."""
    print("DEBUG name:", name)
    print("DEBUG email:", email)
    print("DEBUG consent:", consent)
    print("DEBUG file:", file.filename if file else None)

    try:
        df = _read_any_table(file)

        dataset_name, expected_cols = _detect_dataset(df, MODELS)
        if not dataset_name:
            return JSONResponse(
                {"ok": False, "error": "Could not detect dataset. See /schemas for expected columns."},
                status_code=400,
            )

        bundle = MODELS[dataset_name]
        model = bundle["model"]
        expected = expected_cols or bundle["schema"]["columns"]

        X, missing = _align_columns_flexible(df, expected)
        probs = _predict_probabilities(model, X)
        preds = (probs >= 0.5).astype(int)

        df_out = df.copy()
        df_out["fraud_probability"] = probs.values
        df_out["prediction"] = preds.values

        ts = int(time.time())
        csv_name = f"{dataset_name}_predictions_{ts}.csv"
        xlsx_name = f"{dataset_name}_predictions_{ts}.xlsx"
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        xlsx_path = os.path.join(OUTPUT_DIR, xlsx_name)
        df_out.to_csv(csv_path, index=False)
        try:
            df_out.to_excel(xlsx_path, index=False)
        except Exception:
            xlsx_name = None

        preview_cols = list(df.columns[:6]) + ["fraud_probability", "prediction"]
        preview_cols = [c for c in preview_cols if c in df_out.columns]
        preview = df_out[preview_cols].head(20).to_dict(orient="records")

        meta = {}
        if missing:
            meta["note"] = (
                f"{len(missing)} missing column(s) filled with 0: {missing[:8]}"
                + (" ..." if len(missing) > 8 else "")
            )

        # Email analytics report (if configured)
        await _send_fraud_report(
            email, csv_name, df_out, dataset_name=dataset_name, name=name
        )

        return {
            "ok": True,
            "dataset": dataset_name,
            "name": name,
            "email": email,
            "consent": consent,
            "rows": len(df_out),
            "results": preview,
            "processed_file_csv": csv_name,
            "processed_file_xlsx": xlsx_name,
            **meta,
        }

    except ValueError as ve:
        return JSONResponse({"ok": False, "error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Server error: {str(e)}"}, status_code=500)


@app.get("/download/csv/{filename}")
async def download_csv(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "File not found"}, status_code=404)
    return FileResponse(path, media_type="text/csv", filename=filename)


@app.get("/download/excel/{filename}")
async def download_excel(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"ok": False, "error": "File not found"}, status_code=404)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )

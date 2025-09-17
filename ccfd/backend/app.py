# backend/app.py
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from io import BytesIO

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    status,
    Request,
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

from dotenv import load_dotenv

# auth
from passlib.context import CryptContext
from jose import jwt, JWTError
from pydantic import BaseModel, EmailStr

# plotting & email
import io as pyio
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# local DB/models (expects backend/db.py and backend/models.py)
from db import engine, Base, get_db
import models

# create tables if needed
Base.metadata.create_all(bind=engine)

# -------------------------
# Load environment
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-prod")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# optional email config
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_FROM = os.getenv("MAIL_FROM")
MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_PORT = int(os.getenv("MAIL_PORT", "587") or 587)
MAIL_STARTTLS = os.getenv("MAIL_STARTTLS", "True").lower() == "true"
MAIL_SSL_TLS = os.getenv("MAIL_SSL_TLS", "False").lower() == "true"
EMAIL_ENABLED = all([MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM, MAIL_SERVER])

fm = None
if EMAIL_ENABLED:
    try:
        from fastapi_mail import FastMail, MessageSchema, ConnectionConfig  # type: ignore
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
    except Exception as e:
        print("Warning: fastapi-mail not available or misconfigured:", e)
        fm = None

# -------------------------
# App & CORS
# -------------------------
app = FastAPI(title="Credit Card Fraud Detection API 🚀")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production to your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Constants & paths
# -------------------------
# Try both backend/models and top-level models for ML pickles
MODEL_DIRS = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(BASE_DIR, "..", "models"),
    os.path.join(BASE_DIR, "model"),
]
SCHEMA_FALLBACK_DIR = None
MODEL_DIR = None
for d in MODEL_DIRS:
    if os.path.isdir(d):
        MODEL_DIR = d
        SCHEMA_FALLBACK_DIR = os.path.join(MODEL_DIR, "schemas")
        break

if MODEL_DIR is None:
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    SCHEMA_FALLBACK_DIR = os.path.join(MODEL_DIR, "schemas")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPPORTED_DATASETS = ["creditcard", "merchant"]
DROP_ALWAYS = ["TransactionType", "id", "Location"]

# -------------------------
# Auth utils
# -------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _get_token_from_header(request: Request) -> Optional[str]:
    auth: str = request.headers.get("Authorization") or ""
    if not auth:
        return None
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


async def get_current_user(request: Request, db: Session = Depends(get_db)) -> models.User:
    token = _get_token_from_header(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # we embed user_id in token
        user_id = int(payload.get("user_id"))
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(models.User).filter((getattr(models.User, "id", None) == user_id) | (getattr(models.User, "user_id", None) == user_id)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# -------------------------
# Model loading & helpers
# -------------------------
def _load_sidecar_schema(dataset: str) -> Optional[Dict[str, Any]]:
    if SCHEMA_FALLBACK_DIR and os.path.exists(SCHEMA_FALLBACK_DIR):
        schema_json = os.path.join(SCHEMA_FALLBACK_DIR, f"{dataset}_schema.json")
        if os.path.exists(schema_json):
            with open(schema_json, "r") as f:
                schema = json.load(f)
            if "columns" in schema and isinstance(schema["columns"], list):
                return schema
    return None


def _normalize_bundle(obj: Any, dataset: str) -> Dict[str, Any]:
    if isinstance(obj, dict) and "model" in obj:
        return {"model": obj["model"], "schema": obj.get("schema")}
    sidecar = _load_sidecar_schema(dataset)
    if not sidecar:
        raise ValueError(f"No schema found for {dataset}")
    return {"model": obj, "schema": sidecar}


def _ingest_path(path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    if isinstance(obj, dict) and any(k in SUPPORTED_DATASETS for k in obj.keys()):
        out = {}
        for ds, inner in obj.items():
            if ds in SUPPORTED_DATASETS:
                out[ds] = _normalize_bundle(inner, ds)
        return out
    dataset = os.path.splitext(os.path.basename(path))[0].replace("_model", "")
    if dataset in SUPPORTED_DATASETS:
        return {dataset: _normalize_bundle(obj, dataset)}
    return None


def _detect_dataset(df: pd.DataFrame, bundles: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[List[str]]]:
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
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        else:
            X[col] = X[col].astype(str)
    return X, missing


def _predict_probabilities(model: Any, X: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        m, M = raw.min(), raw.max()
        denom = (M - m) if (M - m) > 1e-12 else 1.0
        return pd.Series((raw - m) / denom, index=X.index)
    return pd.Series(model.predict(X).astype(float), index=X.index)


def _make_plot(df: pd.DataFrame, col: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        sns.histplot(df[col], kde=True, ax=ax)
    except Exception:
        ax.hist(df[col].values)
    buf = pyio.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# -------------------------
# Load ML models
# -------------------------
MODELS: Dict[str, Dict[str, Any]] = {}
if MODEL_DIR and os.path.isdir(MODEL_DIR):
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pkl") or fname.endswith(".joblib"):
            try:
                ingested = _ingest_path(os.path.join(MODEL_DIR, fname))
                if ingested:
                    MODELS.update(ingested)
            except Exception as e:
                print(f"Warning: failed to load model {fname}: {e}")

# try a canonical list if none found
for ds in SUPPORTED_DATASETS:
    if ds not in MODELS:
        maybe = os.path.join(MODEL_DIR, f"{ds}_model.pkl")
        if os.path.exists(maybe):
            MODELS.update(_ingest_path(maybe))

if not MODELS:
    raise FileNotFoundError("No ML models found. Place creditcard_model.pkl, merchant_model.pkl or fraud_detection_model.pkl in models/.")

# -------------------------
# Routes - root & schemas
# -------------------------
@app.get("/")
async def root():
    return {"ok": True, "message": "API running", "models_loaded": list(MODELS.keys())}


@app.get("/schemas")
def schemas():
    return {ds: {"expected_columns": MODELS[ds]["schema"]["columns"]} for ds in MODELS.keys()}


# -------------------------
# Auth endpoints
# -------------------------
@app.post("/register", status_code=201)
def register_user(username: str = Form(...), email: EmailStr = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # check email/username uniqueness
    existing = db.query(models.User).filter((getattr(models.User, "email") == email) | (getattr(models.User, "username") == username)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email or username already registered")
    pwd_hash = get_password_hash(password)
    # fields: support both 'id' and 'user_id' naming in different model versions
    # the provided models.py uses 'id' as primary key; we use constructor fields accordingly
    user_kwargs = {}
    # attempt to set expected field names (username, email, password_hash)
    user_kwargs["username"] = username
    user_kwargs["email"] = email
    # try both password field names
    if "password_hash" in models.User.__dict__:
        user_kwargs["password_hash"] = pwd_hash
    elif "password" in models.User.__dict__:
        user_kwargs["password"] = pwd_hash
    else:
        # fallback to common name
        user_kwargs["password_hash"] = pwd_hash

    user = models.User(**user_kwargs)
    db.add(user)
    db.commit()
    db.refresh(user)

    # extract id for response (support id or user_id)
    uid = getattr(user, "id", None) or getattr(user, "user_id", None)
    return {"ok": True, "user_id": uid, "username": username}


@app.post("/login")
def login(email: EmailStr = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(getattr(models.User, "email") == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    hashed = getattr(user, "password_hash", None) or getattr(user, "password", None)
    if not hashed or not verify_password(password, hashed):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    uid = getattr(user, "id", None) or getattr(user, "user_id", None)
    token = create_access_token({"user_id": int(uid)})
    return {"access_token": token, "token_type": "bearer", "user_id": uid, "username": getattr(user, "username", None)}


# -------------------------
# Protected: history
# -------------------------
@app.get("/history")
def get_history(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Use whichever PK name exists
    user_pk_name = "id" if hasattr(models.User, "id") else "user_id"
    user_pk = getattr(current_user, user_pk_name)
    # Query Uploads - support different Upload field names
    # Our simple models.py uses Upload with user_id, file_name, result_file, created_at
    uploads_q = db.query(models.Upload).filter(getattr(models.Upload, "user_id") == user_pk).order_by(getattr(models.Upload, "created_at").desc())
    uploads = uploads_q.all()
    out = []
    for up in uploads:
        result_file = getattr(up, "result_file", None) or getattr(up, "file_name", None)
        # construct processed filenames if using processed_{upload_id}
        up_id = getattr(up, "id", None) or getattr(up, "upload_id", None) or None
        csv_name = f"processed_{up_id}.csv" if up_id else None
        xlsx_name = f"processed_{up_id}.xlsx" if up_id else None
        csv_exists = os.path.exists(os.path.join(OUTPUT_DIR, csv_name)) if csv_name else False
        xlsx_exists = os.path.exists(os.path.join(OUTPUT_DIR, xlsx_name)) if xlsx_name else False
        out.append({
            "upload_id": up_id,
            "file_name": getattr(up, "file_name", None),
            "result_file": result_file,
            "created_at": getattr(up, "created_at", None) or getattr(up, "created_at", None),
            "processed_csv": csv_name if csv_exists else None,
            "processed_xlsx": xlsx_name if xlsx_exists else None,
        })
    return out


# -------------------------
# Protected: download endpoints
# -------------------------
@app.get("/download/csv/{filename}")
def download_csv(filename: str, current_user: models.User = Depends(get_current_user)):
    safe = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename=safe)


@app.get("/download/excel/{filename}")
def download_excel(filename: str, current_user: models.User = Depends(get_current_user)):
    safe = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=safe)


# -------------------------
# Protected: predict (upload + predict + save)
# -------------------------
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    consent: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    try:
        raw = await file.read()
        fname = (file.filename or "").lower()

        # parse data
        if fname.endswith(".csv"):
            df = pd.read_csv(BytesIO(raw))
        elif fname.endswith(".xls") or fname.endswith(".xlsx"):
            df = pd.read_excel(BytesIO(raw))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type (CSV/XLS/XLSX only)")

        dataset_name, expected_cols = _detect_dataset(df, MODELS)
        if not dataset_name:
            raise HTTPException(status_code=400, detail="Could not detect dataset. See /schemas for expected columns.")

        bundle = MODELS[dataset_name]
        model = bundle["model"]
        expected = expected_cols or bundle["schema"]["columns"]

        X, missing = _align_columns_flexible(df, expected)
        probs = _predict_probabilities(model, X)
        preds = (probs >= 0.5).astype(int)

        df_out = df.copy()
        df_out["fraud_probability"] = probs.values
        df_out["prediction"] = preds.values

        # Save processed files to outputs/ with a name based on timestamp & optional upload id
        # We'll first create DB Upload row (models.Upload expects file_name and result_file)
        # Determine DB user PK
        user_pk = getattr(current_user, "id", None) or getattr(current_user, "user_id", None)

        # Save Upload record
        # Upload model fields in your models.py: id, user_id, file_name, result_file, created_at
        # We'll save result_file as the CSV name (relative path)
        ts = int(time.time())
        # filename base
        base_name = f"{dataset_name}_predictions_{ts}"
        csv_name = f"{base_name}.csv"
        xlsx_name = f"{base_name}.xlsx"
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        xlsx_path = os.path.join(OUTPUT_DIR, xlsx_name)

        # attempt to write files
        try:
            df_out.to_csv(csv_path, index=False)
            try:
                df_out.to_excel(xlsx_path, index=False)
            except Exception:
                xlsx_name = None
        except Exception as e:
            print("Warning: could not write processed files:", e)
            csv_name = None
            xlsx_name = None

        # create upload DB row
        upload_kwargs = {
            "user_id": user_pk,
            "file_name": file.filename,
            "result_file": csv_name or "",
        }
        new_upload = models.Upload(**upload_kwargs)
        db.add(new_upload)
        db.commit()
        db.refresh(new_upload)
        upload_id = getattr(new_upload, "id", None) or getattr(new_upload, "upload_id", None)

        # Optionally, if models include transaction tables (backwards compatibility),
        # try to insert transactions into those tables.
        CreditTxn = getattr(models, "CreditCardTransaction", None)
        MerchantTxn = getattr(models, "MerchantTransaction", None)
        if dataset_name == "merchant" and MerchantTxn is not None:
            for _, row in df.iterrows():
                try:
                    txn_kwargs = {
                        "upload_id": upload_id,
                        "TransactionID": row.get("TransactionID"),
                        "TransactionDate": row.get("TransactionDate"),
                        "Amount": row.get("Amount"),
                        "MerchantID": row.get("MerchantID"),
                        "TransactionType": row.get("TransactionType"),
                        "Location": row.get("Location"),
                    }
                    db.add(MerchantTxn(**txn_kwargs))
                except Exception:
                    continue
            db.commit()
        elif dataset_name == "creditcard" and CreditTxn is not None:
            for _, row in df.iterrows():
                try:
                    txn_kwargs = {"upload_id": upload_id}
                    for v in [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]:
                        txn_kwargs[v] = row.get(v)
                    db.add(CreditTxn(**txn_kwargs))
                except Exception:
                    continue
            db.commit()

        # Try to send an email with a simple analytics plot (best-effort)
        if EMAIL_ENABLED and fm and getattr(current_user, "email", None):
            try:
                img_b64 = _make_plot(df_out, "fraud_probability")
                summary_html = f"""
                <h3>Fraud Analysis Report</h3>
                <p>User: {getattr(current_user, 'username', '')}</p>
                <p>File: {file.filename}</p>
                <p>Rows: {len(df_out)}</p>
                <p>Fraudulent: {(df_out['prediction']==1).sum()}</p>
                <img src="data:image/png;base64,{img_b64}" style="width:600px;max-width:100%;" />
                """
                message = MessageSchema(
                    subject="Your Fraud Analysis Report",
                    recipients=[getattr(current_user, "email")],
                    body=summary_html,
                    subtype="html",
                )
                await fm.send_message(message)
            except Exception as e:
                print("Warning: email send failed:", e)

        # prepare preview for response
        preview_cols = list(df_out.columns[:6]) + ["fraud_probability", "prediction"]
        preview_cols = [c for c in preview_cols if c in df_out.columns]
        preview = df_out[preview_cols].head(20).to_dict(orient="records")

        return {
            "ok": True,
            "upload_db_id": upload_id,
            "dataset": dataset_name,
            "rows": len(df_out),
            "fraudulent": int((df_out["prediction"] == 1).sum()),
            "results_preview": preview,
            "processed_file_csv": csv_name,
            "processed_file_xlsx": xlsx_name,
            "missing_columns_filled": missing,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("Server error in /predict:", exc)
        raise HTTPException(status_code=500, detail="Server error during processing")

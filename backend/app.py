from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os, time

app = FastAPI()

# Allow frontend (index.html) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = joblib.load("fraud_detection_model.pkl")


@app.post("/predict")
async def predict(
    name: str = Form(...),
    email: str = Form(...),
    consent: str = Form(False),
    file: UploadFile = None
):
    if not file:
        return JSONResponse({"error": "No file uploaded"}, status_code=400)

    # Ensure upload + processed folders exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    # Save original file with timestamp to avoid overwrite
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_filename = f"{timestamp}_{file.filename}"
    raw_filepath = os.path.join("uploads", raw_filename)

    with open(raw_filepath, "wb") as f:
        f.write(await file.read())

    try:
        # Load data into pandas
        if raw_filepath.endswith(".csv"):
            df = pd.read_csv(raw_filepath)
        else:
            df = pd.read_excel(raw_filepath)

        # Run predictions
        preds = model.predict(df)
        df["Fraud_Prediction"] = preds

        # Save processed file
        processed_filename = raw_filename.replace(".", "_processed.")
        processed_filepath = os.path.join("processed", processed_filename)
        df.to_csv(processed_filepath, index=False)

        # If consent given, append to master_dataset.csv
        if consent and str(consent).lower() in ["true", "1", "yes", "on"]:
            master_path = "master_dataset.csv"

            # Add user info columns
            df["Uploaded_By"] = name
            df["Email"] = email
            df["Upload_Timestamp"] = timestamp

            if os.path.exists(master_path):
                df.to_csv(master_path, mode="a", header=False, index=False)
            else:
                df.to_csv(master_path, index=False)

        # Convert sample results (first 20 rows) to JSON
        result_json = df.head(20).to_dict(orient="records")

        return {
            "name": name,
            "email": email,
            "consent": bool(consent and str(consent).lower() in ["true", "1", "yes", "on"]),
            "saved_file": raw_filename,
            "processed_file": processed_filename,
            "results": result_json
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download/{filename}")
async def download_file(filename: str):
    filepath = os.path.join("processed", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="text/csv", filename=filename)
    else:
        return JSONResponse({"error": "File not found"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

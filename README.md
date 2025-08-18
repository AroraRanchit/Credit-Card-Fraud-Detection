# 💳 Credit Card Fraud Detection App

A simple web application that allows users to upload their credit card transaction files (CSV/Excel), run them through a trained machine learning model, and view/download the fraud prediction results.  
Users can also choose to allow their data to be stored for future model training.

---

## 🚀 Features
- Upload **CSV/Excel** transaction files.
- Enter **Name** and **Email** before uploading.
- Optional **consent checkbox** to allow data storage for further training.
- ML model (`fraud_detection_model.pkl`) predicts fraud likelihood.
- Display top 20 prediction results in the browser.
- Download processed file (with fraud prediction column).
- Stores:
  - Raw uploaded files in `backend/uploads/`
  - Processed files in `backend/processed/`
  - Consented data in `backend/master_dataset.csv`

---

## 📂 Project Structure
creditcard-fraud-app/
│
├── backend/ # Backend REST API (FastAPI)
│ ├── app.py # Main FastAPI app
│ ├── fraud_detection_model.pkl # Trained ML model
│ ├── uploads/ # Uploaded raw files
│ ├── processed/ # Processed files with predictions
│ ├── master_dataset.csv # (auto-updated with consented data)
│ └── requirements.txt # Python dependencies
│
├── frontend/ # Static frontend (HTML, JS, CSS)
│ └── index.html # Main webpage
│
└── README.md


---

## ⚙️ Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/creditcard-fraud-app.git
cd creditcard-fraud-app/backend


python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


pip install -r requirements.txt


uvicorn app:app --reload


Server runs at http://127.0.0.1:8000

Open Frontend
Open frontend/index.html in your browser.
(Or serve it locally with: cd frontend && python -m http.server 8080)


---

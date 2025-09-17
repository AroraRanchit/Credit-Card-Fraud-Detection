# 💳 Credit Card Fraud Detection (CCFD)

A full-stack machine learning project for detecting fraudulent credit card transactions.  
Users can upload transaction datasets (CSV/Excel), get real-time predictions, view history, and download results.

---

## 📂 Project Structure

```text
ccfd/
├── backend/                  # FastAPI backend
│   ├── app.py                # Main FastAPI app
│   ├── trainer.py            # Model training logic
│   ├── emailsystem.py        # Notification/email system
│   ├── models/               # Pretrained ML models
│   ├── uploads/              # Raw uploaded files
│   ├── outputs/              # Prediction outputs (CSV/Excel)
│   ├── processed/            # Processed files for retraining
│   ├── requirements.txt      # Backend dependencies
│   ├── Dockerfile            # Docker image for backend
│   └── docker-compose.yml    # Docker orchestration
│
├── frontend/                 # Frontend HTML + CSS
│   ├── index.html            # Landing page + upload
│   ├── upload.html           # File upload and results
│   ├── history.html          # Past predictions (requires login)
│   ├── login.html            # Login page
│   ├── signup.html           # Signup page
│   └── style.css             # Shared theme (green/beige pastel)
│
├── models/                   # Schemas + reference models
│   ├── creditcard_model.pkl
│   ├── merchant_model.pkl
│   └── schemas/
│       ├── creditcard_schema.json
│       └── merchant_schema.json
│
└── README.md                 # This file
🚀 Features
Upload datasets (CSV or Excel) → fraud detection in real-time

Download predictions in CSV or Excel format

User login & signup → access your own history

View history of past predictions (per user)

Docker support for easy deployment

Retraining pipeline → extend the model with new data

⚙️ Backend Setup
Navigate to the backend:

bash
Copy code
cd backend
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the server:

bash
Copy code
uvicorn app:app --reload
API will be live at → http://127.0.0.1:8000

🎨 Frontend Setup
The frontend/ folder has static HTML/CSS.
You can serve it directly (open in browser) or integrate with FastAPI static routes.

Main pages:

index.html → Upload & get results

upload.html → Detailed upload

history.html → View past predictions (login required)

login.html → Login

signup.html → Register

style.css → Shared pastel green/beige theme

🐳 Docker Setup
Build and run with Docker:

bash
Copy code
docker-compose up --build
🧠 Model Training
You can retrain models using trainer.py.
It supports schema detection, so new datasets with different columns can still be integrated.

📌 To Do
 Connect login/signup with FastAPI authentication

 Serve frontend through backend routes

 Add role-based access for admin vs users

 Improve history filtering & download options

👨‍💻 Author
Ranchit Arora

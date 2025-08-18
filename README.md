# 💳 Credit Card Fraud Detection

A machine learning–powered web application for detecting fraudulent credit card transactions.  
The project combines a **FastAPI backend** with a **simple HTML frontend** for easy user interaction.  

Users can:
- Upload their transaction data (`CSV` or `Excel`).
- Get real-time predictions on fraudulent vs. legitimate transactions.
- Choose to allow their data to be added for further model training (opt-in).
- Retrain the model on updated datasets when needed.

---

## 🚀 Features
- **Frontend (index.html)**: User-friendly interface for uploading files and viewing results.
- **Backend (FastAPI)**: REST API that processes files and returns fraud predictions.
- **ML Model**: Pre-trained XGBoost model stored as `fraud_detection_model.pkl`.
- **Data Collection**: Stores user uploads in `master_dataset.csv` if consent is given.
- **Retraining**: Supports retraining the ML model with new data.

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
├── backend/
│   ├── app.py                  # FastAPI backend
│   ├── fraud_detection_model.pkl # Trained XGBoost model
│   ├── requirements.txt        # Python dependencies
│   ├── uploads/                # Raw user uploads
│   ├── processed/              # Processed prediction results
│   └── master_dataset.csv      # Growing dataset for retraining
├── frontend/
│   └── index.html              # Frontend webpage
└── README.md                   # Project documentation
⚙️ Installation & Setup
Clone the repository

bash
Copy
Edit
git clone https://github.com/AroraRanchit/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection/backend
Create a virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows use: venv\\Scripts\\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the FastAPI backend

bash
Copy
Edit
uvicorn app:app --reload
The API will start at: http://127.0.0.1:8000

Open the frontend

Open frontend/index.html directly in a browser OR

Serve it locally:

bash
Copy
Edit
cd frontend
python -m http.server
Then visit: http://127.0.0.1:8000

🔎 How It Works
Upload Data: User provides a CSV/Excel file of transactions.

Preprocessing: Missing values are imputed, categorical features encoded.

Prediction: The XGBoost model evaluates transactions.

Results: Fraud likelihood is displayed on the frontend.

Data Consent: If opted-in, the uploaded file is appended to master_dataset.csv.

Retraining: Admins can retrain the model on the updated dataset to improve accuracy.

🧑‍💻 Retraining the Model
If you want to retrain the ML model with new data:

bash
Copy
Edit
cd backend
python retrain.py
This will:

Load master_dataset.csv

Retrain the XGBoost model with Optuna optimization

Save the updated model as fraud_detection_model.pkl

🌐 Deployment
Backend: Can be deployed on Render, Heroku, AWS, or any server that supports FastAPI + Uvicorn.

Frontend: Can be hosted via GitHub Pages, Netlify, or served directly by the backend.

Model File: Large .pkl files (>100 MB) may require Git LFS or cloud storage.

🤝 Contributing
Contributions are welcome!

Fork the repo

Create a feature branch

Submit a Pull Request

📜 License
This project is licensed under the MIT License.

👤 Author
Ranchit Arora
GitHub Profile

pgsql
Copy
Edit

Do you also want me to add **badges** (like Python version, FastAPI, license) at the top so it looks even more professional on GitHub?

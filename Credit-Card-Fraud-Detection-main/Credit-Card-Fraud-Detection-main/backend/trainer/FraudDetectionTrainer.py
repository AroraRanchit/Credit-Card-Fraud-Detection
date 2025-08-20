import pandas as pd
import optuna
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class FraudDetectionTrainer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_and_prepare(self):
        """Load dataset and prepare train-test split."""
        if self.file_path.endswith(".csv"):
            self.df = pd.read_csv(self.file_path)
        else:
            self.df = pd.read_excel(self.file_path)

        if "Class" not in self.df.columns:
            raise ValueError(f"❌ No 'Class' column found in {self.file_path}")

        X = self.df.drop("Class", axis=1)
        y = self.df["Class"]

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

        preprocessor = ColumnTransformer(transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numeric_features)
        ])

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y), preprocessor

    def objective(self, trial, X_train, y_train, preprocessor):
        """Optuna objective for hyperparameter tuning."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        }

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                **params,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",   # GPU compatible
                device="cuda"         # Use CUDA GPU
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
        return scores.mean()

    def train(self, n_trials=100):
        """Train with Optuna hyperparameter tuning and evaluate best model."""
        (X_train, X_test, y_train, y_test), preprocessor = self.load_and_prepare()

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, preprocessor),
                       n_trials=n_trials)

        print(f"✅ {self.file_path} → Best Optuna Accuracy: {study.best_value:.4f}")
        print(f"🎯 Best Params: {study.best_params}")

        # Retrain with best params
        best_model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                **study.best_params,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",
                device="cuda"
            ))
        ])

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"   Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # Save trained model + schema
        schema = {
            "columns": list(X_train.columns),
            "dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()}
        }

        bundle = {
            "model": best_model,
            "schema": schema
        }

        joblib.dump(bundle, "fraud_detection_model.pkl")
        print("💾 Model + schema saved as fraud_detection_model.pkl")


if __name__ == "__main__":
    file_path = r"C:\Users\arora\Downloads\creditcardtrainer\creditcard_2023.csv"  # change if needed
    trainer = FraudDetectionTrainer(file_path)
    trainer.train(n_trials=100)

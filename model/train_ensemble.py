"""
Train a RandomForest-based fraud detection model.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/claims.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def main():
    df = pd.read_csv(DATA_PATH)
    y = (df["fraud_flag"] == "Y").astype(int)

    feature_cols = [
        "customer_age",
        "claim_amount",
        "num_previous_claims",
        "claim_type",
        "has_police_report",
    ]
    X = df[feature_cols]

    numeric = ["customer_age", "claim_amount", "num_previous_claims"]
    categorical = ["claim_type", "has_police_report"]

    preprocessor = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )

    pipe = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipe, MODEL_DIR / "fraud_ensemble.joblib")
    print("✅ Saved model to models/fraud_ensemble.joblib")


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


FEATURES = [
    "GrLivArea",
    "TotalBsmtSF",
    "OverallQual",
    "YearBuilt",
    "YearRemodAdd",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "GarageCars",
    "TotRmsAbvGrd",
    "Fireplaces",
]
TARGET = "SalePrice"


def preprocess_train(
    train_csv_path, scaler_path="scaler.joblib", test_size=0.2, random_state=42
):
    df = pd.read_csv(train_csv_path)
    df = df[FEATURES + [TARGET]].dropna()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    print(f"Scaler sauvegardé : {scaler_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test


def preprocess_test(test_csv_path, scaler_path="scaler.joblib"):
    df = pd.read_csv(test_csv_path)
    df = df[FEATURES].dropna()

    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(df)

    return X_test_scaled

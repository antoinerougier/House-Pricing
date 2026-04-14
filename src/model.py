import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import os


def train_model(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }

    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    return grid_search.best_estimator_


def save_model(model, path="model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="model.joblib"):
    return joblib.load(path)

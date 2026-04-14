import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }

    return results


def print_results(results):
    print("\n===== Résultats d'évaluation =====")
    print(f"  MAE  : {results['MAE']}")
    print(f"  RMSE : {results['RMSE']}")
    print(f"  R²   : {results['R2']}")
    print("==================================\n")

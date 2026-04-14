from src.preprocessing import preprocess_train
from src.model import train_model, save_model
from src.evaluation import evaluate_model, print_results


def main():
    train_csv = "data/train.csv"  # Change selon ton chemin

    print("Prétraitement...")
    X_train, X_test, y_train, y_test = preprocess_train(
        train_csv, scaler_path="scaler.joblib"
    )

    print("Entraînement du modèle (GridSearch)...")
    model = train_model(X_train, y_train)

    print("Évaluation...")
    results = evaluate_model(model, X_test, y_test)
    print_results(results)

    save_model(model, path="model.joblib")


if __name__ == "__main__":
    main()

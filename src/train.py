import argparse
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_iris_csv, features_labels, evaluate
from models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Treinar e avaliar modelos no dataset Iris")
    parser.add_argument("--data-path", type=str, default="data/iris.csv",
                        help="Caminho para o CSV do dataset Iris")
    parser.add_argument("--model", type=str, required=True,
                        help="Modelo: dt, knn, mlp, nb, perc")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporcao do conjunto de teste (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Seed para reproducao")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Caminho para salvar o modelo treinado (ex.: models/model_dt.pkl)")
    parser.add_argument("--standardize", action="store_true",
                        help="Padroniza features (recomendado p/ knn, mlp, perc)")
    return parser.parse_args()

def main():
    args = parse_args()

    df = load_iris_csv(args.data_path)
    X, y = features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, shuffle=True, random_state=args.random_state
    )

    model = get_model(args.model, random_state=args.random_state)

    steps = []
    if args.standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", model))
    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics_text = evaluate(y_test, y_pred)
    print(metrics_text)

    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        joblib.dump(pipeline, args.save_model)
        print(f"\nModelo salvo em: {args.save_model}")

if __name__ == "__main__":
    main()

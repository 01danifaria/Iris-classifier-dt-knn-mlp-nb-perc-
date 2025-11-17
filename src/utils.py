import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

SPECIES_TO_INT = {
    "Iris-setosa": 1,
    "Iris-versicolor": 2,
    "Iris-virginica": 3,
}
INT_TO_SPECIES = {v: k for k, v in SPECIES_TO_INT.items()}
INT_TO_HUMAN = {1: "setosa", 2: "versicolor", 3: "virginica"}

def load_iris_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=[
        "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
    ])
    df["species"] = df["species"].replace(SPECIES_TO_INT)
    return df

def features_labels(df: pd.DataFrame):
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]
    return X, y

def evaluate(y_true, y_pred) -> str:
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return f"Acuracia: {acc:.4f}\n\nRelatorio de classificacao:\n{report}"

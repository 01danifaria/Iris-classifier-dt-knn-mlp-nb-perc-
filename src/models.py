from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

def get_model(model_key: str, random_state: int = 42):
    key = model_key.lower()
    if key == "dt":
        return DecisionTreeClassifier(random_state=random_state)
    if key == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if key == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000,
                             random_state=random_state)
    if key == "nb":
        return GaussianNB()
    if key == "perc":
        return Perceptron(random_state=random_state)
    raise ValueError(f"Modelo desconhecido: {model_key}. Use dt, knn, mlp, nb, perc.")

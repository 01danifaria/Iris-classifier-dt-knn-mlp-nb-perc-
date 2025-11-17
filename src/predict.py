import argparse
import joblib
import numpy as np
from utils import INT_TO_HUMAN

def parse_args():
    parser = argparse.ArgumentParser(description="Predicao com modelo Iris treinado")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Caminho para o arquivo .pkl do modelo (Pipeline)")
    parser.add_argument("--sepal-length", type=float, required=True, help="Comprimento da sepala (cm)")
    parser.add_argument("--sepal-width", type=float, required=True, help="Largura da sepala (cm)")
    parser.add_argument("--petal-length", type=float, required=True, help="Comprimento da petala (cm)")
    parser.add_argument("--petal-width", type=float, required=True, help="Largura da petala (cm)")
    return parser.parse_args()

def main():
    args = parse_args()
    pipeline = joblib.load(args.model_path)

    sample = np.array([[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]])
    pred_class = int(pipeline.predict(sample)[0])

    print(f"Especie prevista: {INT_TO_HUMAN[pred_class]} (classe={pred_class})")

if __name__ == "__main__":
    main()

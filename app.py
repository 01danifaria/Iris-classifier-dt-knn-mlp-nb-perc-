import streamlit as st
import joblib
import numpy as np

# Carregar modelo treinado (exemplo: Decision Tree salvo em models/model_dt.pkl)
# Certifique-se de treinar e salvar o modelo antes de rodar este app
MODEL_PATH = "models/model_dt.pkl"
model = joblib.load(MODEL_PATH)

# Dicionário de espécies
INT_TO_HUMAN = {1: "setosa", 2: "versicolor", 3: "virginica"}

# Título da aplicação
st.title("🌸 Classificação de Espécies de Iris")
st.write("Insira as medidas da flor para prever a espécie.")

# Inputs do usuário
sepal_length = st.number_input("Comprimento da sépala (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Largura da sépala (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Comprimento da pétala (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Largura da pétala (cm)", 0.0, 10.0, 0.2)

# Botão de classificação
if st.button("Classificar"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_class = int(model.predict(sample)[0])
    st.success(f"🌼 Espécie prevista: **{INT_TO_HUMAN[pred_class]}** (classe={pred_class})")

# Iris-classifier-dt-knn-mlp-nb-perc-

# Projeto de Classificação de Espécies de Íris

Este projeto tem como objetivo aplicar técnicas de **aprendizado de máquina supervisionado** para a classificação de flores da espécie *Iris*, utilizando o conjunto de dados clássico proposto por Ronald Fisher (1936). O trabalho foi desenvolvido por Daniela Maria Barbosa Faria, como parte da disciplina **Engenharias de Computação (AG2)**.

Link do vídeo pelo OneDrive: https://1drv.ms/v/c/238659ffa195e117/EXpJX1EOG45Ep7044QZBylQBR7xAWSN6kDhtBiU0K-NqvA
---

## Metodologia

1. **Aquisição dos dados**  
   O dataset foi obtido do [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

2. **Pré-processamento**  
   - Conversão da coluna *species* de valores textuais para inteiros:  
     - *Iris-setosa* → 1  
     - *Iris-versicolor* → 2  
     - *Iris-virginica* → 3  

3. **Divisão dos dados**  
   - 80% para treinamento  
   - 20% para teste  
   - Embaralhamento realizado para evitar viés na separação.

4. **Modelos de classificação disponíveis**  
   - Árvore de decisão (Decision Tree)  
   - k-Vizinhos mais próximos (k-Nearest Neighbors)  
   - Perceptron multicamadas (MLP)  
   - Naïve Bayes  
   - Perceptron simples  

5. **Treinamento e avaliação**  
   - Execução dos modelos com parâmetros padrão.  
   - Avaliação por meio de métricas de acurácia e relatório de classificação.  

6. **Predição de novas amostras**  
   - Interface em linha de comando para inserção de medidas arbitrárias.  
   - Retorno da espécie prevista com base no modelo treinado.

---

## Execução

### Instalação das dependências
```bash
pip install -r requirements.txt
```

### Baixar o dataset Iris
```bash
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" -o data/iris.csv
```

### Exemplo com Decision Tree:
```bash
python src/train.py --model dt --save-model models/model_dt.pkl
```

### Exemplo com kNN e MLP:
```bash
python src/train.py --model knn --standardize --save-model models/model_knn.pkl
python src/train.py --model mlp --standardize --save-model models/model_mlp.pkl
```

### Predição via Linha de Comando
```bash
python src/predict.py --model-path models/model_dt.pkl \
  --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

### Interface - Instalar Streamlit
``` bash
pip install streamlit
streamlit run app.py
```





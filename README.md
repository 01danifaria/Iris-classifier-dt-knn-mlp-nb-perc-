# Iris-classifier-dt-knn-mlp-nb-perc-

# Projeto de Classificação de Espécies de Íris

Este projeto tem como objetivo aplicar técnicas de **aprendizado de máquina supervisionado** para a classificação de flores da espécie *Iris*, utilizando o conjunto de dados clássico proposto por Ronald Fisher (1936). O trabalho foi desenvolvido por Daniela Maria Barbosa Faria, como parte da disciplina **Engenharias de Computação (AG2)**.

---

## Estrutura do Projeto

- **data/iris.csv** → Conjunto de dados contendo 150 amostras, cada uma com quatro atributos numéricos e a respectiva espécie.
- **src/train.py** → Script para treinamento e avaliação dos modelos.
- **src/predict.py** → Script para predição de novas amostras fornecidas pelo usuário.
- **src/models.py** → Definição e seleção dos algoritmos de classificação.
- **src/utils.py** → Funções auxiliares para carregamento de dados, mapeamento de classes e métricas.
- **notebooks/exploracao.ipynb** → Análise exploratória inicial do conjunto de dados.
- **requirements.txt** → Lista de dependências necessárias para execução.
- **video_script.md** → Roteiro sugerido para apresentação em vídeo.

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


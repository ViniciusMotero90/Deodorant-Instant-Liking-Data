from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=dados.feature_names) 
y = pd.Series(dados.target)

valores_C = np.array([95,96,97,98,99,100,101,102,103,104,105])
regularizacao = ['l1', 'l2']
valores_grid = {'C': valores_C, 'penalty': regularizacao, 'solver': ['liblinear']}

modelo = LogisticRegression(penalty='l1',max_iter=1000)

# Avaliação do modelo
grid_regressao_logistica = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
grid_regressao_logistica.fit(x, y)

print(f"Melhor acurácia: {grid_regressao_logistica.best_score_}")
print(f"Parâmetro C: {grid_regressao_logistica.best_estimator_.C}")
print(f"Regularização: {grid_regressao_logistica.best_estimator_.penalty}")

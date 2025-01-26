import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

# Configurações para exibição
pd.set_option('display.max_columns', 64)
pd.set_option('display.max_rows', 64)

# Leitura do dataset
data = pd.read_csv('Data_train_reduced.csv')

# Remoção de colunas desnecessárias
columns_to_drop = [
    'q8.20', 'q8.18', 'q8.17', 'q8.8', 'q8.9', 'q8.10', 'q8.2', 
    'Respondent.ID', 'Product', 'q1_1.personal.opinion.of.this.Deodorant'
]
data.drop(columns=columns_to_drop, axis=1, inplace=True)

# Tratamento de valores nulos
data['q8.12'] = data['q8.12'].fillna(data['q8.12'].median())
data['q8.7'] = data['q8.7'].fillna(data['q8.7'].median())

# Separação de variáveis independentes e dependentes
y = data['Instant.Liking']
x = data.drop('Instant.Liking', axis=1)

valores_C = np.array([0.01,0.5,1,2,3,5,10,20,50,100])
regularizacao = ['l1','l2']
valores_grid = {'C': valores_C, 'penalty':regularizacao}

# Modelo de Regressão Logística
modelo = LogisticRegression(penalty='l2', solver='liblinear')

# Avaliação do modelo
grid_regressao_logistica = GridSearchCV(estimator=modelo,param_grid=valores_grid,cv=5)
grid_regressao_logistica.fit(x,y)

print(f"Melhor acurácia: {grid_regressao_logistica.best_score_}")
print(f"Parâmetro C: {grid_regressao_logistica.best_estimator_.C}")
print(f"Regularização: {grid_regressao_logistica.best_estimator_.penalty}")
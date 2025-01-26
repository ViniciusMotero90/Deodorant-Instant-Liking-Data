import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

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

# Configuração de validação cruzada
stratifiedkfold = StratifiedKFold(n_splits=5)

# Modelo de Regressão Logística
modelo = LogisticRegression(penalty='l2', solver='liblinear')

# Avaliação do modelo
resultado = cross_val_score(modelo, x, y, cv=stratifiedkfold)

print(f"Acurácia média: {resultado.mean():.4f}")

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

#Importando o dataset de custo individual do seguro saúde 
datasetSeguroSaude = pd.read_csv("insurance.csv")

#Transfomando as variáveis categóricas em variáveis numéricas utilizando o LabelEncoder
label_encoder = LabelEncoder()
datasetSeguroSaude["smoker"] = label_encoder.fit_transform(datasetSeguroSaude["smoker"])
datasetSeguroSaude["sex"] = label_encoder.fit_transform(datasetSeguroSaude["sex"])
datasetSeguroSaude["region"] = label_encoder.fit_transform(datasetSeguroSaude["region"])

#Separando os dados em TREINO e TESTE

#Criando uma coluna de estrato combinando as características índice de massa corporal, idade e indicador de fumante
datasetSeguroSaude["cat1"] = pd.cut(datasetSeguroSaude["bmi"], bins=4, labels=False) # agrupando bmi em 4 categorias
datasetSeguroSaude["estrato"] = datasetSeguroSaude["cat1"].astype(str) + '_' + datasetSeguroSaude["smoker"].astype(str) 

#Separando os dados, com estratificação, para evitar enviesamento
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(datasetSeguroSaude, datasetSeguroSaude["estrato"]):
    conjunto_treino = datasetSeguroSaude.loc[train_index]
    conjunto_teste = datasetSeguroSaude.loc[test_index]

#Separando a TARGET dos conjuntos de dados 
target_treino = conjunto_treino["charges"].copy() 
target_teste = conjunto_teste["charges"].copy() 

#Removendo as colunas não relevantes para o algoritmo
for set_ in (conjunto_treino, conjunto_teste):
    set_.drop("cat1", axis=1, inplace=True)
    set_.drop("estrato", axis=1, inplace=True)
    set_.drop("charges", axis=1, inplace=True)

#Removendo as colunas incluidas no arquivo original
datasetSeguroSaude.drop("cat1", axis=1, inplace=True)
datasetSeguroSaude.drop("estrato", axis=1, inplace=True)

#Treinando o modelo de regressão linear
regressao_linear = LinearRegression()
regressao_linear.fit(conjunto_treino, target_treino)

#Testando o algoritmo
predicao = regressao_linear.predict(conjunto_teste)

accuracy = accuracy_score(target_teste, predicao)
print(f'Acurácia: {accuracy:.2f}')


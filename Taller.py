# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:17:57 2022

@author: Flia. Menco Ariza
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

url = 'bank-full.csv'

#--------------------------DATASET BANK FULL-----------------------------------

data = pd.read_csv(url)


#Remplazos
data.default.replace(['no','yes'], [0,1], inplace= True)
data.job.replace(['blue-collar','management','technician','admin.','services','retired','self-employed','entrepreneur','unemployed','housemaid','student','unknown'], [0,1,2,3,4,5,6,7,8,9,10,11], inplace= True)
data.marital.replace(['married','single','divorced'], [0,1,2], inplace= True)
data.education.replace(['secondary','tertiary','primary','unknown'], [0,1,2,3], inplace= True)
data.housing.replace(['no','yes'], [0,1], inplace= True)
data.loan.replace(['no','yes'], [0,1], inplace= True)
data.contact.replace(['cellular','unknown','telephone'], [0,1,2], inplace= True)
data.poutcome.replace(['unknown','failure','other','success'], [0,1,2,3], inplace= True)
data.y.replace(['no','yes'], [0,1], inplace= True)

#---------------------------------------------------------------------------------------------------
rangos = [18,25,40,60,100]
nombres = ['1','2','3','4',]
data.age = pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0,how='any', inplace=True)

#---------------------------------------------------------------------------------------------------
#Columnas Innecesarias
data.drop(['balance', 'day', 'month', 'duration','campaign','pdays','previous'], axis= 1, inplace = True)

#Dividir Data
data_train = data[:30000]
data_test = data[30000:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)


#REGRESION LOGISTICA ----------------------------------------------------------

#MODELO
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

#ENTRENAMIENTO
logreg.fit(x_train,y_train)

# METRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')



# MAQUINA DE SOPORTE VECTORIAL-------------------------------------------------

# MODELO
svc = SVC(gamma='auto')

# ENTRENAMIENTO
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# ARBOL DE DECISIÓN------------------------------------------------------------

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')



# RANDOM FOREST------------------------------------------------------------

# Seleccionar un modelo
forest = RandomForestClassifier()

# Entreno el modelo
forest.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('RANDOM FOREST')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {forest.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {forest.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {forest.score(x_test_out, y_test_out)}')


# NAIVE BAYES------------------------------------------------------------


# Seleccionar un modelo
nayve = GaussianNB()

# Entreno el modelo
nayve.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('NAYVE BAYES')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {nayve.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {nayve.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {nayve.score(x_test_out, y_test_out)}')
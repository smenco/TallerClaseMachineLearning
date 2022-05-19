# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:49:12 2022

@author: Flia. Menco Ariza
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

url = 'weatherAUS.csv'

#--------------------------DATASET weatherAUS----------------------------------

data = pd.read_csv(url)

#Remplazos
rangos = [-8.0,0,10,20,30,40]
nombres = ['1','2','3','4','5']
data.MinTemp = pd.cut(data.MinTemp, rangos, labels=nombres)


rangos2 = [0,10,20,30,40,50]
nombres2 = ['1','2','3','4','5']
data.MaxTemp = pd.cut(data.MaxTemp, rangos2, labels=nombres2)


rangos3 = [-1,50,100,150,200,250]
nombres3 = ['1','2','3','4','5']
data.Rainfall = pd.cut(data.Rainfall, rangos3, labels=nombres3)


rangos4 = [-1,20,40,60,80,100]
nombres4 = ['1','2','3','4','5']
data.Evaporation = pd.cut(data.Evaporation, rangos4, labels=nombres4)


rangos5 = [-1,5,10,15]
nombres5 = ['1','2','3']
data.Sunshine = pd.cut(data.Sunshine, rangos5, labels=nombres5)



rangos6 = [0,30,60,90,130]
nombres6 = ['1','2','3','4']
data.WindGustSpeed = pd.cut(data.WindGustSpeed, rangos6, labels=nombres6)


rangos7 = [0,20,40,60,80]
nombres7 = ['1','2','3','4']
data.WindSpeed9am = pd.cut(data.WindSpeed9am, rangos7, labels=nombres7)


rangos8 = [0,20,40,60,80]
nombres8 = ['1','2','3','4']
data.WindSpeed3pm = pd.cut(data.WindSpeed3pm, rangos8, labels=nombres8)


rangos9 = [-1,20,40,60,80,101]
nombres9 = ['1','2','3','4','5']
data.Humidity9am = pd.cut(data.Humidity9am, rangos9, labels=nombres9)


rangos10 = [-1,20,40,60,80,101]
nombres10 = ['1','2','3','4','5']
data.Humidity3pm = pd.cut(data.Humidity3pm, rangos10, labels=nombres10)


rangos11 = [980,1000,1050]
nombres11 = ['1','2']
data.Pressure9am = pd.cut(data.Pressure9am, rangos11, labels=nombres11)


rangos12 = [970,1000,1040]
nombres12 = ['1','2']
data.Pressure3pm = pd.cut(data.Pressure3pm, rangos12, labels=nombres12)


rangos13 = [-0.5,0,20,40]
nombres13 = ['1','2','3']
data.Temp9am = pd.cut(data.Temp9am, rangos13, labels=nombres13)


rangos14 = [0,20,40,50]
nombres14 = ['1','2','3']
data.Temp3pm = pd.cut(data.Temp3pm, rangos14, labels=nombres14)


data['RainToday'].replace(['No', 'Yes'], [0, 1], inplace=True)
data['RainTomorrow'].replace(['No', 'Yes'], [0, 1], inplace=True)


data.dropna(axis=0,how='any', inplace=True)


#---------------------------------------------------------------------------------------------------
#Columnas Innecesarias
data.drop(['Date','Location','WindGustDir','WindDir9am','WindDir3pm','RISK_MM'], axis= 1, inplace = True)


#---------------------------------------------------------------------------------------------------
# partir la data en dos

data_train = data[:40000]
data_test = data[40000:]


x = np.array(data_train.drop(['RainTomorrow'], 1))
y = np.array(data_train.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], 1))
y_test_out = np.array(data_test.RainTomorrow)


# REGRESIÓN LOGÍSTICA CON VALIDACIÓN CRUZADA

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

print('*'*50)
print('Regresión Logística Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion regresion logistica")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')




# ARBOL DE DECISIÓN CON VALIDACION CRUZADA------------------------------------------------------------

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
# MÉTRICAS

for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)


# MÉTRICAS

print('*'*50)
print('Arbol de decision con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion arbol de decision")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')



# RANDOM FOREST------------------------------------------------------------

# Seleccionar un modelo
forest = RandomForestClassifier()

# Entreno el modelo
forest.fit(x_train, y_train)

# MÉTRICAS

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
# MÉTRICAS

for train, test in kfold.split(x, y):
    forest.fit(x[train], y[train])
    scores_train_train = forest.score(x[train], y[train])
    scores_test_train = forest.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = forest.predict(x_test_out)


# MÉTRICAS

print('*'*50)
print('Random forest con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {forest.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion Random forest")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')


# NAIVE BAYES CON VALIDACION CRUZADA------------------------------------------------------------


# Seleccionar un modelo
nayve = GaussianNB()

# Entreno el modelo
nayve.fit(x_train, y_train)

# MÉTRICAS

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
# MÉTRICAS

for train, test in kfold.split(x, y):
    nayve.fit(x[train], y[train])
    scores_train_train = nayve.score(x[train], y[train])
    scores_test_train = nayve.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = nayve.predict(x_test_out)


# MÉTRICAS

print('*'*50)
print('Nayve bayes Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {nayve.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion Nayve")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')



# MAQUINA DE SOPORTE VECTORIAL CON VALIDACION CRUZADA-------------------------------------------------

# MODELO
svc = SVC(gamma='auto')

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []

# ENTRENAMIENTO

for train, test in kfold.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)


# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial con Validación cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusión
print(f'Matriz de confusión: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion maquina de soporte vectorial")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisión: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')
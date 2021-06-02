# -*- coding: utf-8 -*-
"""
TRABAJO 3 Clasificación
Nombre Estudiante: Mario Carmona Segovia
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

# Fijamos la semilla
np.random.seed(1)


# Funcion para leer los datos
def readDataTXT(file, numColumX, numColumY):
    # Leemos los ficheros
    File = open(file)
    reader = csv.reader(File, delimiter=' ')

    data = []
    for row in reader:
        data.append(row)

    data = np.array(data, np.float64)

    y = data[:, numColumY]
    x = data[:, numColumX]
	
    return x, y

# Función para separar los datos de entrada y salida
def separarTrainTest(X, Y, porcentaje_test):
    # El porcentaje de test, indica el porcentaje de elementos que forman parte del test
    
    clases, counts = np.unique(Y, return_counts=True)
    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    for i in clases:
        indices = (Y == i)
        
        datosX = X[np.ravel(indices), :]
        datosY = Y[np.ravel(indices), :]
        
        numElemTrain = (int)(datosX.shape[0] * (1-porcentaje_test))
        
        X_train.append( datosX[np.arange(0,numElemTrain,1), :] )
        Y_train.append( datosY[np.arange(0,numElemTrain,1), :] )
        
        X_test.append( datosX[np.arange(numElemTrain,datosX.shape[0],1), :] )
        Y_test.append( datosY[np.arange(numElemTrain,datosX.shape[0],1), :] )
    
    X_train = np.array(X_train).reshape((-1,X.shape[1]))
    Y_train = np.array(Y_train).reshape((-1,Y.shape[1]))
    X_test = np.array(X_test).reshape((-1,X.shape[1]))
    Y_test = np.array(Y_test).reshape((-1,Y.shape[1]))

    return X_train, X_test, Y_train, Y_test

# Función que muestra dos tablas con la separación de los datos de training y test
def mostrarSeparacionDatos(X, Y, X_train, Y_train, X_test, Y_test):
    numElemX = X.shape[0]
    numElemXtrain = X_train.shape[0]
    numElemXtest = X_test.shape[0]

    numElemY = Y.shape[0]
    numElemYtrain = Y_train.shape[0]
    numElemYtest = Y_test.shape[0]

    print('\t\tTamaño\tPorcentaje')
    print('------- -------  ----------')
    print('X\t\t{}\t{}%'.format(numElemX, (numElemX/numElemX)*100))
    print('X_train\t{}\t{}%'.format(numElemXtrain,
          round((numElemXtrain/numElemX)*100, 2)))
    print('X_test\t{}\t{}%'.format(numElemXtest,
          round((numElemXtest/numElemX)*100, 2)))

    print('\n\t\tTamaño\tPorcentaje')
    print('------- -------  ----------')
    print('Y\t\t{}\t{}%'.format(numElemY, (numElemY/numElemY)*100))
    print('Y_train\t{}\t{}%'.format(numElemYtrain,
          round((numElemYtrain/numElemY)*100, 2)))
    print('Y_test\t{}\t{}%'.format(numElemYtest,
          round((numElemYtest/numElemY)*100, 2)))
    
    clases, countsClases = np.unique(Y, return_counts=True)
    
    print('\nClase\tTotal\tTrain\tTest')
    print('-----   -----    -----    ----')
    for i in clases:
        datosTrain = (Y_train == i)
        unique, counts = np.unique(datosTrain, return_counts=True)
        dic = dict(zip(unique, counts))
        numTrain = dic[True]
        datosTest = (Y_test == i)
        unique, counts = np.unique(datosTest, return_counts=True)
        dic = dict(zip(unique, counts))
        numTest = dic[True]
        
        print('{}\t\t{}\t\t{}\t\t{}'.format((int)(i), countsClases[(int)(i-1)], numTrain, numTest))
        
        
# Función para obtener las correlaciones entre las características
def obtenerMatrizCorrelaciones(X):
    return np.corrcoef(X, rowvar=False)

        
# Función que muestra la matriz de correlaciones
def mostrarMatrizCorrelaciones(matrizCorrelaciones):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

    sns.heatmap(
        matrizCorrelaciones,
        square    = True,
        ax        = ax
    )
    
    ax.tick_params(labelsize = 3)

    plt.ylabel('Características')
    plt.xlabel('Características')
    plt.title('Matriz de correlaciones')
    plt.show(fig)


numColumX = np.arange(0, 48, 1)
numColumY = np.arange(48, 49, 1)

# Lectura de los datos
X, Y = readDataTXT('datos/Sensorless_drive_diagnosis.txt', numColumX, numColumY)

print('\n\tSeparación de los datos en training y test\n')

# Separación de los datos en train y test
porcentaje_test = 0.2

X_train, X_test, Y_train, Y_test = separarTrainTest(X, Y, porcentaje_test)

mostrarSeparacionDatos(X, Y, X_train, Y_train, X_test, Y_test)

input("\n--- Pulsar tecla para continuar ---\n")

# Análisis de las correlaciones entre los atributos
matrizCorrelaciones = obtenerMatrizCorrelaciones(X_train)

minimo = np.absolute(matrizCorrelaciones).min()
media = np.absolute(matrizCorrelaciones).mean()

print('\tAnálisis de las relaciones entre variables\n')

print('Mínimo valor absoluto de correlación: {}\n'.format(minimo))

print('Media del valor absoluto de correlación: {}'.format(media))

mostrarMatrizCorrelaciones(matrizCorrelaciones)

input("\n--- Pulsar tecla para continuar ---\n")

def eliminarDatosSinVari(X, Y):
    media = []
    desviacion = []
    datosEliminados = []

    # Comprobar que datos eliminar y calcular media y
    # desviación de los que no son eliminados
    for i in np.arange(0, X.shape[1], 1):
        unique = np.unique(X[:, i])
        if unique.shape[0] == 1:
            datosEliminados.append(i)
        else:
            media.append(np.mean(X[:, i]))
            desviacion.append(np.std(X[:, i]))

    # Eliminar características sin variabilidad
    np.delete(X, datosEliminados)
    np.delete(Y, datosEliminados)
    
    return X, Y


def normalizacionDatos(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_norma_train = scaler.transform(X_train)
    X_norma_test = scaler.transform(X_test)

    return X_norma_train, X_norma_test


def mostrarRangoDatos(X, title):
    minimos = []
    maximos = []
    medias = []

    for i in np.arange(0, X.shape[1], 1):
        minimos.append(X[:, i].min())
        maximos.append(X[:, i].max())
        medias.append(X[:, i].mean())

    fig = plt.figure()

    plt.plot(np.arange(0, X.shape[1], 1), minimos, c='blue', label='Mínimos')
    plt.plot(np.arange(0, X.shape[1], 1), maximos, c='red', label='Máximos')
    plt.plot(np.arange(0, X.shape[1], 1), medias, c='green', label='Medias')

    plt.title(title)
    plt.xlabel('Características')
    plt.ylabel('Valores')

    plt.legend()

    plt.show(fig)


def preprocesadoDatos(X_train, Y_train, X_test, Y_test):
    # Eliminar las características sin variabilidad
    numCaracIni = X_train.shape[1]
    X_train, Y_train = eliminarDatosSinVari(X_train, Y_train)
    X_test, Y_test = eliminarDatosSinVari(X_test, Y_test)

    print('- Eliminación de características sin variabilidad\n')
    print('Se han eliminado {} características de los datos de train\n'.format(
        numCaracIni-X_train.shape[1]))
    print('Se han eliminado {} características de los datos de test\n'.format(
        numCaracIni-X_test.shape[1]))

    # Normalización de los datos

    print('- Normalización de los datos\n')

    print('Datos antes de ser normalizados:')

    mostrarRangoDatos(X_train, 'Datos sin normalizar')

    X_train, X_test = normalizacionDatos(X_train, X_test)

    print('\nDatos después de ser normalizados:')

    mostrarRangoDatos(X_train, 'Datos normalizados')
    
    # Datos extremos

    print('\n- Datos extremos\n')
    
    numEjemplosIniTrain = X_train.shape[0]
    numEjemplosIniTest = X_test.shape[0]
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(X_train)
    LOF_score = clf.negative_outlier_factor_
    LOF_score = np.sort(LOF_score)[::-1]
    cuartil1 = LOF_score[int(0.25*LOF_score.shape[0])]
    cuartil3 = LOF_score[int(0.75*LOF_score.shape[0])]
    distanInterCuartil = cuartil3 - cuartil1
    constante = 10
    X_train = X_train[clf.negative_outlier_factor_ >= (cuartil3 + constante*distanInterCuartil)]
    Y_train = Y_train[clf.negative_outlier_factor_ >= (cuartil3 + constante*distanInterCuartil)]
    
    clf.fit_predict(X_test)
    LOF_score = clf.negative_outlier_factor_
    LOF_score = np.sort(LOF_score)[::-1]
    cuartil1 = LOF_score[int(0.25*LOF_score.shape[0])]
    cuartil3 = LOF_score[int(0.75*LOF_score.shape[0])]
    distanInterCuartil = cuartil3 - cuartil1
    X_test = X_test[clf.negative_outlier_factor_ >= (cuartil3 + constante*distanInterCuartil)]
    Y_test = Y_test[clf.negative_outlier_factor_ >= (cuartil3 + constante*distanInterCuartil)]
    
    print('Porcentaje de datos extremos en el train: {}%'.format( round(((numEjemplosIniTrain - X_train.shape[0]) / numEjemplosIniTrain) * 100,2) ))
    print('Número de ejemplos que son outliers en el train: {}\n'.format( numEjemplosIniTrain - X_train.shape[0] ))
    
    print('Porcentaje de datos extremos en el test: {}%'.format( round(((numEjemplosIniTest - X_test.shape[0]) / numEjemplosIniTest) * 100,2) ))
    print('Número de ejemplos que son outliers en el test: {}'.format( numEjemplosIniTest - X_test.shape[0] ))
    
    return X_train, Y_train, X_test, Y_test


print('\tPreprocesado de los datos\n')

X_train, Y_train, X_test, Y_test = preprocesadoDatos(X_train, Y_train, X_test, Y_test)

X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis=1)
X_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis=1)

input("\n--- Pulsar tecla para continuar ---\n")

def mostrarVarianzaExplicada(pca):
    fig = plt.figure()
    
    for i in np.arange(0,pca.n_components_,1):
        plt.bar(i, pca.explained_variance_ratio_[i])
        
    plt.ylabel('Varianza explicada')
    plt.xlabel('Características')
    plt.title('Gráfica de la varianza explicada')
        
    plt.show(fig)
    
def mostrarVarianzaExplicadaAcu(pca):
    fig = plt.figure()
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    indices = (cumsum >= 0.99)

    unique, index = np.unique(indices, return_index=True)    
    dic = dict(zip(unique, index))

    plt.plot(np.arange(0,pca.n_components_,1), cumsum, markersize=3.5, linestyle='-', marker='o')
    plt.plot([dic[True],dic[True]], [0,1], linestyle='--', c='red')
    
    plt.ylabel('Varianza explicada acumulada')
    plt.xlabel('Características')
    plt.title('Gráfica de la varianza explicada acumulada')
        
    plt.show(fig)
    
    print('\n - Con las {} primeras características se tiene más de un 99% de la varianza explicada'.format(dic[True]))


pca = PCA(n_components=None)
pca.fit(X_train)

print('\tAnálisis de la varianza explicada por cada característica\n')

mostrarVarianzaExplicada(pca)

mostrarVarianzaExplicadaAcu(pca)

input("\n--- Pulsar tecla para continuar ---\n")

print('\tEstimación de hiperparámetros\n')

print('- Modelo SGD\n')

max_iter = np.ceil(10**6 / X_train.shape[0])

estimator = SGDClassifier(loss='log', max_iter=max_iter)

model = OneVsRestClassifier(estimator=estimator)



parametros = {'estimator__penalty': ['l1', 'l2'], 'estimator__alpha': [1, 0.1, 0.01]
              ,'estimator__learning_rate': ['constant'],'estimator__eta0': [0.01, 0.015, 0.02]}

gridSGD = GridSearchCV(model, parametros, cv=5, scoring='accuracy', return_train_score=False)
gridSGD.fit(X_train, np.ravel(Y_train))

penaltySGD = gridSGD.best_estimator_._first_estimator.penalty
alphaSGD = gridSGD.best_estimator_._first_estimator.alpha
etaSGD = gridSGD.best_estimator_._first_estimator.eta0
best_scoreSGD = gridSGD.best_score_

print('Los parámetros de la mejor hipótesis con Regresión logística son:\n\n\t- penalty: {}\n\t- alpha: {}\n\t- eta: {}'.format(penaltySGD, alphaSGD, etaSGD))

print('\nError de la mejor hipótesis: {}'.format(round(best_scoreSGD,2)))

input("\n--- Pulsar tecla para continuar ---\n")

print('- Modelo Perceptron\n')

estimator = Perceptron()

model = OneVsRestClassifier(estimator=estimator)

parametros = {'estimator__penalty': ['l1', 'l2'], 'estimator__alpha': [1, 0.1, 0.01]
              ,'estimator__eta0': [0.01, 0.015, 0.02]}

gridPerceptron = GridSearchCV(model, parametros, cv=5, scoring='accuracy', return_train_score=False)
gridPerceptron.fit(X_train, np.ravel(Y_train))

penaltyPerceptron = gridPerceptron.best_estimator_._first_estimator.penalty
alphaPerceptron = gridPerceptron.best_estimator_._first_estimator.alpha
etaPerceptron = gridPerceptron.best_estimator_._first_estimator.eta0
best_scorePerceptron = gridPerceptron.best_score_

print('Los parámetros de la mejor hipótesis con Perceptron son:\n\n\t- penalty: {}\n\t- alpha: {}\n\t- eta: {}'.format(penaltyPerceptron, alphaPerceptron, etaPerceptron))

print('\nError de la mejor hipótesis: {}'.format(round(best_scorePerceptron,2)))

input("\n--- Pulsar tecla para continuar ---\n")

modelo = None

if(best_scoreSGD > best_scorePerceptron):
    modelo = OneVsRestClassifier(SGDClassifier(loss='log', max_iter=max_iter, penalty=penaltySGD, alpha=alphaSGD, learning_rate='constant', eta0=etaSGD))
    print('La mejor hipótesis es la mejor hipótesis obtenida con SGD')
else:
    modelo = OneVsRestClassifier(Perceptron(penalty=penaltyPerceptron, alpha=alphaPerceptron, eta0=etaPerceptron))
    print('La mejor hipótesis es la mejor hipótesis obtenida con Perceptron')

input("\n--- Pulsar tecla para continuar ---\n")

modelo.fit(X_train, np.ravel(Y_train))

predicciones = modelo.predict(X_test)
E_test = accuracy_score(Y_test, predicciones)

print('El error de test de la hipótesis final es {}'.format(round(E_test,2)))

n_classes = 11

y_score = modelo.decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

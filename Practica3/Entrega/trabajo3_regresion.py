# -*- coding: utf-8 -*-
"""
TRABAJO 3 Regresion
Nombre Estudiante: Mario Carmona Segovia
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


# Fijamos la semilla
np.random.seed(1)


# Funcion para leer los datos
def readDataCSV(file, numColumX, numColumY):
    # Leemos los ficheros
    File = open(file)
    reader = csv.reader(File, delimiter=',')

    data = []
    for row in reader:
        data.append(row)

    # Eliminar cabecera del CSV
    data.pop(0)

    data = np.array(data, np.float64)

    y = data[:, numColumY]
    x = data[:, numColumX]

    return x, y

# Función para separar los datos de entrada y salida
def separarTrainTest(X, Y, porcentaje_test):
    # El porcentaje de test, indica el porcentaje de elementos que forman parte del test
    
    # Cálculo del número de elementos de training
    numElemTrain = X.shape[0] - X.shape[0] * porcentaje_test

    # Los primeros elementos son para training
    elemTrain = np.arange(0, numElemTrain, 1, dtype=np.int64)
    # Los últimos elementos son para test
    elemTest = np.arange(numElemTrain, X.shape[0], 1, dtype=np.int64)
    
    # Separar los datos de entrada y salida para training y test
    X_train = X[elemTrain, :]
    Y_train = Y[elemTrain, :]
    X_test = X[elemTest, :]
    Y_test = Y[elemTest, :]

    return X_train, X_test, Y_train, Y_test

# Función para obtener las correlaciones entre las características
def obtenerMatrizCorrelaciones(X):
    return np.corrcoef(X, rowvar=False)

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
    print('X_test\t{}\t\t{}%'.format(numElemXtest,
          round((numElemXtest/numElemX)*100, 2)))

    print('\n\t\tTamaño\tPorcentaje')
    print('------- -------  ----------')
    print('Y\t\t{}\t{}%'.format(numElemY, (numElemY/numElemY)*100))
    print('Y_train\t{}\t{}%'.format(numElemYtrain,
          round((numElemYtrain/numElemY)*100, 2)))
    print('Y_test\t{}\t\t{}%'.format(numElemYtest,
          round((numElemYtest/numElemY)*100, 2)))
    
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




# Selección de la columnas para los datos de entrada y salida
numColumX = np.arange(0, 81, 1)
numColumY = np.arange(81, 82, 1)

# Lectura de los datos
X, Y = readDataCSV('datos/train.csv', numColumX, numColumY)

print('\n\tSeparación de los datos en training y test\n')

# Separación de los datos en train y test
porcentaje_test = 0.2

X_train, X_test, Y_train, Y_test = separarTrainTest(X, Y, porcentaje_test)

# Visualizar la separación
mostrarSeparacionDatos(X, Y, X_train, Y_train, X_test, Y_test)

input("\n--- Pulsar tecla para continuar ---\n")

# Análisis de las correlaciones entre los atributos
matrizCorrelaciones = obtenerMatrizCorrelaciones(X_train)

# Obtener el mínimo y la media de las correlaciones
minimo = np.absolute(matrizCorrelaciones).min()
media = np.absolute(matrizCorrelaciones).mean()

print('\tAnálisis de las relaciones entre variables\n')

print('Mínimo valor absoluto de correlación: {}\n'.format(round(minimo,7)))

print('Media del valor absoluto de correlación: {}'.format(round(media,4)))

# Visualizar la matriz de correlaciones
mostrarMatrizCorrelaciones(matrizCorrelaciones)

input("\n--- Pulsar tecla para continuar ---\n")

# Función para eliminar las características sin variabilidad
def eliminarDatosSinVari(X, Y):
    datosEliminados = []

    # Seleccionar que datos eliminar comprobando la variabilidad de cada característica
    for i in np.arange(0, X.shape[1], 1):
        unique = np.unique(X[:, i])
        if unique.shape[0] == 1:
            datosEliminados.append(i)

    # Eliminar características sin variabilidad
    np.delete(X, datosEliminados)
    np.delete(Y, datosEliminados)
    
    return X, Y

# Función para normalizar el valor de las características dentro del rango [0,1]
def normalizacionDatos(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_norma_train = scaler.transform(X_train)
    # Se normalizan los datos de test con la función ajusta con los datos de training
    X_norma_test = scaler.transform(X_test)

    return X_norma_train, X_norma_test

# Función para mostrar todas las características con sus rangos
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

# Función para preprocesar los datos de training y test
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
    
    # Eliminación de los datos extremos

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

# Añadir una columna de unos para el w_0 de la función
X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis=1)
X_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis=1)

input("\n--- Pulsar tecla para continuar ---\n")

# Función que muestra la varianza explicada por cada característica por separado
def mostrarVarianzaExplicada(pca):
    fig = plt.figure()
    
    for i in np.arange(0,pca.n_components_,1):
        plt.bar(i, pca.explained_variance_ratio_[i])
        
    plt.ylabel('Varianza explicada')
    plt.xlabel('Características')
    plt.title('Gráfica de la varianza explicada')
        
    plt.show(fig)
    
# Función que muestra la varianza acumulada
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

print('- Modelo Ridge\n')

model = Ridge(solver='svd')
parametros = {'alpha': [0.1, 0.01]}

gridRidge = GridSearchCV(model, parametros, cv=10, scoring='neg_root_mean_squared_error', return_train_score=False)
gridRidge.fit(X_train, Y_train)

alphaRidge = gridRidge.best_estimator_.alpha
best_scoreRidge = -gridRidge.best_score_

print('Los parámetros de la mejor hipótesis con Ridge son:\n\n\t- alpha: {}'.format(alphaRidge))

print('\nError de la mejor hipótesis: {}'.format(round(best_scoreRidge,2)))

input("\n--- Pulsar tecla para continuar ---\n")

print('- Modelo SGD\n')

max_iter = np.ceil(10**6 / X_train.shape[0])*2

model = SGDRegressor()
parametros = {'loss': ['squared_loss'], 'penalty': ['l1', 'l2'], 'alpha': [0.001, 0.0001]
              ,'learning_rate': ['constant'], 'eta0': [0.01, 0.015, 0.02], 'max_iter': [max_iter]}

gridSGD = GridSearchCV(model, parametros, cv=10, scoring='neg_root_mean_squared_error', return_train_score=False)
gridSGD.fit(X_train, np.ravel(Y_train))

penaltySGD = gridSGD.best_estimator_.penalty
alphaSGD = gridSGD.best_estimator_.alpha
etaSGD = gridSGD.best_estimator_.eta0
n_iterSGD = gridSGD.best_estimator_.n_iter_
max_iterSGD = gridSGD.best_estimator_.max_iter
best_scoreSGD = -gridSGD.best_score_

print('Los parámetros de la mejor hipótesis con SGD son:\n\n\t- penalty: {}\n\t- alpha: {}\n\t- eta: {}\n\t- n_iter: {}\n\t- max_iter: {}'.format(penaltySGD, alphaSGD, etaSGD, n_iterSGD, max_iterSGD))

print('\nError de la mejor hipótesis: {}'.format(round(best_scoreSGD,2)))

input("\n--- Pulsar tecla para continuar ---\n")

modelo = None

if(best_scoreRidge < best_scoreSGD):
    modelo = Ridge(alpha=alphaRidge, max_iter=1000)
    print('La mejor hipótesis es la mejor hipótesis obtenida con Ridge')
else:
    modelo = SGDRegressor(penalty=penaltySGD, alpha=alphaSGD, eta0=etaSGD, max_iter=max_iter)
    print('La mejor hipótesis es la mejor hipótesis obtenida con SGD')

input("\n--- Pulsar tecla para continuar ---\n")

modelo.fit(X_train, np.ravel(Y_train))

predicciones = modelo.predict(X_test)
E_test = mean_squared_error(Y_test, predicciones, squared=False)

media = Y_test.mean()

E_test_base = mean_squared_error(Y_test, np.repeat(media, Y_test.shape[0]), squared=False)

print('El error de test de la hipótesis final es {}'.format(round(E_test,2)))
print('El error de test base de la hipótesis final es {}'.format(round(E_test_base,2)))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

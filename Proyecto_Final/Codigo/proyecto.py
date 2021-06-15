"""
Proyecto final
Nombre Estudiantes: Mario Carmona Segovia
                    Francisco Jose Aparicio Martos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
import time


# Función para leer los datos
def readData(fichero):
    datos = pd.read_csv(fichero)
    data = np.array(datos)
    return data

# Función para estandarizar los datos de entrada
def estandarizar(xn,x):
    scaler = StandardScaler()
    scaler.fit(xn)
    new_x = scaler.transform(x)
    return new_x

# Función para mostrar la media y varianza de las características
def mostrarRangoDatos(X, title, escala_log):
    medias = []
    varianzas = []

    for i in np.arange(0, X.shape[1], 1):
        varianzas.append(X[:, i].var())
        medias.append(X[:, i].mean())

    plt.plot(np.arange(0, X.shape[1], 1), medias, c='blue', label='Medias')
    plt.plot(np.arange(0, X.shape[1], 1), varianzas, c='red', label='Varianzas')

    plt.title(title)
    plt.xlabel('Características')
    plt.ylabel('Valores')
    
    if(escala_log):
        plt.yscale('symlog')

    plt.legend()

    plt.show()

# Función para realizar el preprocesado de los datos de entrada
def preprocesar(xn,x):
    #en el preprocesado estandarizamos y añadimos la columna de 1's
    new_x = estandarizar(xn,x)
    #new_x = np.append(np.ones([np.size(new_x,0),1]),new_x,axis=1)

    return new_x

# Función para mostrar la puntuación de la distintas combinaciones que se
# realizan en cross-validation
def plot_bars_charts(bars,models,num_bars,title):
    plt.figure(figsize=(20,10))
    bars = np.array(bars,np.float64)
    plt.bar(models,bars)
    plt.xlabel('Configuracion de hiperparametros')
    plt.ylabel("Balanced_accuracy")
    plt.title(title)
    plt.show()
    
# Función para buscar el mejor modelo de un algoritmo dado un grid de parámetros
def buscarMejorModelo(modelo,param_grid):
    return GridSearchCV(modelo,param_grid,cv=5,scoring="balanced_accuracy",return_train_score=True)

# Función para obtener el mejor modelo con el algoritmo regresión logística
def RL(x,y):

    sgd = SGDClassifier()
    #uso gridSeachCv para la busqueda de los mejores parámetros para sgdRegressor
    sgd_params = [{'loss':['log'],'penalty':['l2','l1'],'alpha':[0.1,0.01,0.001],'fit_intercept':[True],
                   'max_iter':[10**3,10**4],'shuffle':[True],'random_state':[24],'learning_rate':['constant'],
                   'eta0':[0.1,0.01,0.001],'early_stopping': [False,True], 'validation_fraction':[0.1],
                   'n_iter_no_change':[5], 'class_weight': ['balanced'],'warm_start':[False],'average':[False]}]
    
    mejor_sgd = buscarMejorModelo(sgd,sgd_params)
    mejor_sgd.fit(x,y)
    
    print("Los mejores parametros para la regresión logística son:\n")
    print(mejor_sgd.best_params_,"\n")
    
    print("Combinaciones estudiadas\n")
    res = mejor_sgd.cv_results_
    bars = []
    models = []
    index = []
    i = 0
    for mean_score,params in zip(res["mean_test_score"],res["params"]):
        i+=1
        models.append(str(i));
        index.append("alpha: " + str(params['alpha'])+" eta0: "+ str(params['eta0'])+" max_iter: "+ str(params['max_iter'])+" penalty: "+params['penalty'])  
        bars.append(mean_score)
        
        
    plot_bars_charts(bars,models,i,"Estudio de parametros de regresion logistica")
    for j,l in zip(range(1,len(index)+1),index):
        print(j,":",l)
        
    return mejor_sgd.best_estimator_

# Función para obtener el mejor modelo con el algoritmo multi layer perceptron
def MLP(x,y):
    #usamos la función BalancedBaggingClassifier devido a que por defecto mlp no tiene disponible la opción de trabajar con clases desbalanceadas
    mlp = BalancedBaggingClassifier(base_estimator=MLPClassifier(),
                            sampling_strategy='auto',
                            replacement=False,
                            random_state=0)
    
    #print(mlp.get_params().keys())
    #hidden_layers_sizes: número de neuronas por capa oculta
    #activation: función de activación usada en cada neurona
    #solver: algortimo usado en la red neuronal, usaremos sgd al igual que se enseño en teoría
    #alpha: parámetro que multiplica el término de regularización, su tamaño indica la intensidad de la regularización aplicada
    #learning_rate: tipo de comportamiento del learning_rate a lo largo de la ejecución
    #learning_rate_init: valor inicial del learning_rate, si es muy grande se tiende a diverger y si es muy pequeño se tarda en converger
    #power_t se usa solo cuando learning_rate = invsclaing, por lo tanto no lo usamos
    #mas_iter: número de épocas máxima que puede completar el objetivo antes de su finalización
    #shuffle: indica si se barajan los datos para hacer el gradiente descendente estocástico
    #random_state: semilla para la generación aleatoria
    #tol: tolerancia de la optimización, cuando se han realizado n_iter_no_change iteraciones sin cambio el algoritmo para
    #verbose: true o false si queremos más o menos información acerca del modelo
    #warm_start: se indica si se va a iniciar con un solución previa.
    #momentum: parámetro que multiplica el término de momentum
    #nevsterovs_momentum: indica si usamos el momementum en el aprendizaje, este se basa en usar los pasos tomados previamente para actualizar el vector de pesos, se suelen conseguir buenos resultados en la práctica
    #early_stopping: indicamos si se usa un porcentaje de la muestra para usarla en el proceso de validación en el early stopping
    #validation_fracton: tamaño del conjunto de validación
    #beta1,beta2,epsilon no lo usamos ya que no usamos el solver adam
    #n_iter_no_change: número de epocas que no se actualizan las épocas, usado para el early_stopping
    #max_fun no se usa en el solver sgd
    mlp_params = [{'base_estimator__hidden_layer_sizes':[(100,50,),(50,100,),(50,50,),(100,100,)],
                   'base_estimator__activation':['relu'],'base_estimator__solver':['sgd'],'base_estimator__alpha':[0.1,0.01,0.001],
                   'base_estimator__learning_rate':['constant'], 'base_estimator__learning_rate_init':[0.1,0.01,0.001],
                   'base_estimator__max_iter':[1000], 'base_estimator__shuffle':[True], 'base_estimator__random_state':[24],
                   'base_estimator__tol':[10**(-4)], 'base_estimator__warm_start':[False], 'base_estimator__momentum':[0.9],
                   'base_estimator__nesterovs_momentum':[True],
                   'base_estimator__early_stopping':[True],'base_estimator__validation_fraction':[0.1],
                   'base_estimator__n_iter_no_change':[10]}]
    
    mejor_mlp = buscarMejorModelo(mlp,mlp_params)
    mejor_mlp.fit(x,y)
    
    print("Los mejores parametros para la mlp son:\n")
    print(mejor_mlp.best_params_,"\n")
    
    print("Combinaciones estudiadas\n")
    res = mejor_mlp.cv_results_
    bars = []
    models = []
    index = []
    i = 0
    for mean_score,params in zip(res["mean_test_score"],res["params"]):
        i+=1
        models.append(str(i));
        index.append("units_per_hidden_layers: " + str(params['base_estimator__hidden_layer_sizes'])
                     + " alpha: " + str(params['base_estimator__alpha'])
                     +" learning_rate_init: "+ str(params['base_estimator__learning_rate_init'])
                     +" max_iter: "+ str(params['base_estimator__max_iter'])
                     +" activation_function: " + str(params['base_estimator__activation'])
                     )  
        bars.append(mean_score)
        
        
    plot_bars_charts(bars,models,i,"Estudio de parametros de mlp")
    for j,l in zip(range(1,len(index)+1),index):
        print(j,":",l)
        
    return mejor_mlp.best_estimator_

# Función para obtener el mejor modelo con el algoritmo support vector machine
def SVM(x,y):
    svm = SVC()
    #C: constante que multiplica al término de regularización, al ser una svm la regularización usada es la l2
    #kernel: kernel que se va a aplicar 
    #degree: grados del kernel polinomial, solo es válido si se usa kernel polinomial
    #gamma: el coeficiente para los kernels rbd, polinomial y sigmoidal
    #coef0: termino independiente de la función kernel
    #shrinking: indica si se usa la heurística shrinking
    #probability: indicamos si se quiere hacer el calculo de las probabilidades estimadas
    #tol: tolerancia de nuestro algoritmo
    #cache_size: se indica la cantidad de memoria cache que usará el algoritmo
    #class_weight: se indica si las clases estan desbalanceadas
    #verbose: opción para que el resultado contenga más o menos detalles
    #max_iter: número máximo de iteraciones de nuestro algoritmo
    #decision_function_shape: forma en la que se realiza la clasificación de las distintas clases, como es clasificación binaria se ignora
    #break_ties: solo se usa en clasificación con más de 2 clases
    #random_state: semilla que se usará en la generación de número aleatorios, solo se usa si se realiza el calculo de probabilidades
    
    svm_params_poly = [{'C':[2,1], 'kernel':['poly'], 'degree':[2,3,5],
                   'gamma':['scale','auto'], 'coef0':[0,1], 'shrinking':[True,False], 'probability':[False],
                   'tol':[0.001], 'cache_size':[500], 'class_weight': ['balanced'],
                   'max_iter':[-1]}]
    
    svm_params_rbf = [{'C':[2,1], 'kernel':['rbf'],
                   'gamma':['scale','auto'], 'coef0':[0,1], 'shrinking':[True,False], 'probability':[False],
                   'tol':[0.001], 'cache_size':[500], 'class_weight': ['balanced'],
                   'max_iter':[-1]}]
    
    mejor_svm_poly = buscarMejorModelo(svm,svm_params_poly)
    mejor_svm_poly.fit(x,y)
    
    mejor_svm_rbf = buscarMejorModelo(svm,svm_params_rbf)
    mejor_svm_rbf.fit(x,y)
    
    print("Los mejores parametros para la svm son:\n")
    print(mejor_svm_poly.best_params_,"\n")
    
    print("Combinaciones estudiadas\n")
    res = mejor_svm_poly.cv_results_
    bars = []
    models = []
    index = []
    i = 0
    for mean_score,params in zip(res["mean_test_score"],res["params"]):
        i+=1
        models.append(str(i));
        index.append("C: " + str(params['C'])
                     + " kernel: " + str(params['kernel'])
                     +" degree: "+ str(params['degree'])
                     +" gamma: "+ str(params['gamma'])
                     +" coef0: " + str(params['coef0'])
                     +" shrinking: " + str(params['shrinking'])
                     )  
        bars.append(mean_score)
        
    res = mejor_svm_rbf.cv_results_
        
    for mean_score,params in zip(res["mean_test_score"],res["params"]):
        i+=1
        models.append(str(i));
        index.append("C: " + str(params['C'])
                     + " kernel: " + str(params['kernel'])
                     +" gamma: "+ str(params['gamma'])
                     +" coef0: " + str(params['coef0'])
                     +" shrinking: " + str(params['shrinking'])
                     )  
        bars.append(mean_score)
        
        
    plot_bars_charts(bars,models,i,"Estudio de parametros de svm")
    for j,l in zip(range(1,len(index)+1),index):
        print(j,":",l)
        
    return mejor_svm_poly.best_estimator_

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
    print('X\t\t{}\t\t{}%'.format(numElemX, (numElemX/numElemX)*100))
    print('X_train\t{}\t\t{}%'.format(numElemXtrain,
          round((numElemXtrain/numElemX)*100, 2)))
    print('X_test\t{}\t\t{}%'.format(numElemXtest,
          round((numElemXtest/numElemX)*100, 2)))

    print('\n\t\tTamaño\tPorcentaje')
    print('------- -------  ----------')
    print('Y\t\t{}\t\t{}%'.format(numElemY, (numElemY/numElemY)*100))
    print('Y_train\t{}\t\t{}%'.format(numElemYtrain,
          round((numElemYtrain/numElemY)*100, 2)))
    print('Y_test\t{}\t\t{}%'.format(numElemYtest,
          round((numElemYtest/numElemY)*100, 2)))
    
    clases, countsClases = np.unique(Y, return_counts=True)
    
    print('\nClase\tTotal\t\tTrain\t\tTest')
    print('-----   ----------   ----------   --------')
    for i in clases:
        numTotal = Y[Y == i].shape[0]
        numTrain = Y_train[Y_train == i].shape[0]
        numTest = Y_test[Y_test == i].shape[0]
        
        proporcionTotal = numTotal / Y.shape[0]
        proporcionTrain = numTrain / Y_train.shape[0]
        proporcionTest = numTest / Y_test.shape[0]
        
        print('{}\t\t{} ({})\t{} ({})\t{} ({})'.format(i, numTotal, round(proporcionTotal,2), numTrain, round(proporcionTrain,2), numTest, round(proporcionTest,2)))

# Función para mostrar una gráfica 3D con los datos reducidos
def mostrarDatosReducidos(X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    indices_clase1 = (Y == 'M')
    indices_clase2 = (Y == 'B')
    ax.scatter(X[indices_clase1,0], X[indices_clase1,1], X[indices_clase1,2], c='blue', marker='o', label='Clase Malignos')
    ax.scatter(X[indices_clase2,0], X[indices_clase2,1], X[indices_clase2,2], c='red', marker='o', label='Clase Benignos')

    plt.title('Visión 3D de los datos de training reducidos')
    ax.set_xlabel('Textura media')
    ax.set_ylabel('Peor área')
    ax.set_zlabel('Peor suavidad')

    plt.show()


##################################################################
##################################################################

# Lectura de los datos
x = readData("datos/data.csv")

# Eliminamos una columna de valores nan que se añade al leer los datos del archivo CSV
x = x[:,:-1]

# Eliminamos los id de los ejemplos
x = x[:,1:]

# Obtenemos los datos de salida
y = x[:,0]

# Obtenemos los datos de entrada
x = x[:,1:]

x = x.astype(np.float64)


# Separación de los datos en training y test

print('Separación de los datos en training y test:\n')

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, stratify = y,random_state=42)

mostrarSeparacionDatos(x, y, x_train, y_train, x_test, y_test)

input("\n--- Pulsar tecla para continuar ---\n")


# Preprocesado de los datos de entrada

print('Preprocesado de los datos:\n')

print('\t- Datos de training')

mostrarRangoDatos(x_train, 'Datos de training sin estandarizar', True)
xn_train = preprocesar(x_train,x_train)
mostrarRangoDatos(xn_train[:,1:], 'Datos de training estandarizados', False)

xn_test = preprocesar(x_train,x_test)

input("\n--- Pulsar tecla para continuar ---\n")


print('Visualización de los datos de training reducidos:\n')

indices = [1,23,24]
mostrarDatosReducidos(xn_train[:,indices], y_train)

input("\n--- Pulsar tecla para continuar ---\n")


# Experimento sobre la codificación de las clases

y_train_codi = np.array(y_train)

y_train_codi[y_train_codi == 'M'] = 1
y_train_codi[y_train_codi == 'B'] = -1
y_train_codi = y_train_codi.astype(np.float64)

rl_estimator = SGDClassifier(loss='log', class_weight='balanced')

svc_estimator = SVC(kernel='rbf', class_weight = 'balanced')

mlp_balanced = BalancedBaggingClassifier(base_estimator=MLPClassifier(max_iter = 1000),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

print('Experimento con la codificación de las clases:\n')

print('\n\tRL:\n')

scores = cross_val_score(rl_estimator,xn_train,y_train,cv=10,scoring="balanced_accuracy")
print("Eout (sin clases codificadas) = ",np.mean(scores))
scores = cross_val_score(rl_estimator,xn_train,y_train_codi,cv=10,scoring="balanced_accuracy")
print("Eout (con clases codificadas) = ",np.mean(scores))

print('\n\tMLP:\n')

scores = cross_val_score(mlp_balanced,xn_train,y_train,cv=10,scoring="balanced_accuracy")
print("Eout (sin clases codificadas) = ",np.mean(scores))
scores = cross_val_score(mlp_balanced,xn_train,y_train_codi,cv=10,scoring="balanced_accuracy")
print("Eout (con clases codificadas) = ",np.mean(scores))

print('\n\tSVC:\n')

scores = cross_val_score(svc_estimator,xn_train,y_train,cv=10,scoring="balanced_accuracy")
print("Eout (sin clases codificadas) = ",np.mean(scores))
scores = cross_val_score(svc_estimator,xn_train,y_train_codi,cv=10,scoring="balanced_accuracy")
print("Eout (con clases codificadas) = ",np.mean(scores))

input("\n--- Pulsar tecla para continuar ---\n")

# Indices de las características que se eligen en la reducción
indices = [1,23,24]
x_train_reducido = xn_train[:,indices]
x_test_reducido = xn_test[:,indices]

# Elección del mejor modelo con cada algoritmo, con reducción de características
# y sin la reducción de características

print("\nElección del mejor modelo para mlp sin usar la reducción de variables\n")
start = time.time()
mlp_mejor = MLP(xn_train,y_train)
end = time.time()
scores = cross_val_score(mlp_mejor,xn_train,y_train,cv=10,scoring="balanced_accuracy")
print("\nEout mejor mlp (sin reducción) = ",np.mean(scores))

print("Se tarda en la selección del mejor modelo : ", end-start, " segundos")

print("\nElección del mejor modelo para mlp con la reducción de variables\n")
start = time.time()
mlp_mejor_reducido = MLP(x_train_reducido,y_train)
end = time.time()
scores = cross_val_score(mlp_mejor,x_train_reducido,y_train,cv=10,scoring="balanced_accuracy")
print("\nEout mejor mlp (con reducción) = ",np.mean(scores))

print("Se tarda en la selección del mejor modelo : ", end-start, " segundos")

input("\n--- Pulsar tecla para continuar ---\n")


print("\nElección del mejor modelo para svm sin usar la reducción de variables\n")
start = time.time()
svm_mejor = SVM(xn_train,y_train)
end = time.time()
scores = cross_val_score(svm_mejor,xn_train,y_train,cv=10,scoring="balanced_accuracy")
print("\nEout mejor svm (sin reducción) = ",np.mean(scores))

print("Se tarda en la selección del mejor modelo : ", end-start, " segundos")

print("\nElección del mejor modelo para svm con la reducción de variables\n")
start = time.time()
svm_mejor_reducido = SVM(x_train_reducido,y_train)
end = time.time()
scores = cross_val_score(svm_mejor,x_train_reducido,y_train,cv=10,scoring="balanced_accuracy")
print("\nEout mejor svm (con reducción) = ",np.mean(scores))

print("Se tarda en la selección del mejor modelo : ", end-start, " segundos")

input("\n--- Pulsar tecla para continuar ---\n")


print("\nElección del mejor modelo para regresión logística sin usar la reducción de variables\n")
start = time.time()
rl_mejor = RL(xn_train,y_train)
end = time.time()
scores = cross_val_score(rl_mejor,xn_train,y_train,cv=10,scoring="balanced_accuracy")
print("\nEout mejor regresión logística (sin reducción) = ",np.mean(scores))

print("Se tarda en la selección del mejor modelo : ", end-start, " segundos")

print("\nElección del mejor modelo para regrsión logística con la reducción de variables\n")
start = time.time()
rl_mejor_reducido = RL(x_train_reducido,y_train)
end = time.time()
scores = cross_val_score(rl_mejor,x_train_reducido,y_train,cv=10,scoring="balanced_accuracy")
print("\nEout mejor regresión logística (con reducción) = ",np.mean(scores))

print("Se tarda en la selección del mejor modelo : ", end-start, " segundos")

input("\n--- Pulsar tecla para continuar ---\n")


print('Visualización de la matriz de confusión:\n')

svm_mejor.fit(xn_train,y_train)

fig = plot_confusion_matrix(svm_mejor, xn_test, y_test, display_labels=['Maligno', 'Benigno'], cmap="Blues")
fig.ax_.set_title('Matriz de confusión')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


print('Visualización de la curva de ROC:\n')

plot_roc_curve(svm_mejor, xn_test, y_test)
plt.plot([0,1],[0,1], linestyle='--', c='red', label='Clasificador aleatorio (AUC = 0.5)')
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


print("Calculo de la cota de nuestro mejor modelo:\n")

svm_mejor.fit(xn_train,y_train)
pred = svm_mejor.predict(xn_test)

Etest = 1 - balanced_accuracy_score(y_test,pred)
Eout = Etest + np.sqrt((1/(2*np.size(xn_test,0)))*np.log((2)/0.05))
print("Eout <= ", Etest + np.sqrt((1/(2*np.size(xn_test,0)))*np.log((2)/0.05)))
print("Por lo que la precisión de nuestra modelo será mayor o igual a ", 1 - Eout)



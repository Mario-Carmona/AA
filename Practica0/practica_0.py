# -*- coding: utf-8 -*-
"""
Práctica 0

Mario Carmona Segovia - Grupo 1
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


####################################################
###                    Parte 1                   ###
####################################################

# Para esta práctica como datos vamos a coger la base de datos
# de iris que se encuentra en scikit-learn.
#
# Esta base de datos está formada por diferentes lirios clasificados
# en 3 clases de lirios (Setosa, Versicolor, Virginica). Las
# características de cada lirio son, longitud del sépalo, 
# tamaño del sépalo, longitud del pétalo y tamaño del pétalo.

# Cargamos la BD de iris en una variable, esta contendrá una
# array que tendrá como columnas las características de los lirios
# y como filas los distintos lirios examinados.
iris = datasets.load_iris()

# Obtener los datos de entrada (características) quedándonos sólo
# con la primera y tercera columna
X = iris.data[:,::2]

# Obtener las clases de cada lirio examinado
y = iris.target

# Obtener el nombre de las clases
clases = iris.target_names

# Obtener las características seleccionadas
carac = np.array(iris.feature_names)[::2]

# Mostrar información sobre los datos
print('## Primera parte ##\n')
texto = '→ Clases de lirios: '
for i in clases:
    texto = texto + '"' + i + '"' + ' '    
texto += '\n'
texto += '→ Características de los lirios: '
for i in iris.feature_names:
    texto = texto + '"' + i + '"' + ' '   
texto += '\n'
texto += '→ Num. lirios examinados: ' + str(X[:,0].size)
print(texto)

# Crear gráfico con los datos, para esta parte he partido
# de los ejemplos sobre iris que se nos proporciona en la
# documentación

# Creo la primera figura
fig1 = plt.figure(1, figsize=(8,6))

# Creamos un vector de colores para indicar los color con los
# que queremos mostrar los puntos del gráfico
colores = np.array(['orange', 'black', 'green'])

# Calcular el límite de los valores de cada eje en la gráfica
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

# Calculo cuantos elementos hay en cada clase, lo hago de esta
# forma porque en X los elementos de cada clase están guardados
# de forma consecutiva
unique, counts = np.unique(y, return_counts=True)

# Creamos unas variables auxiliares para poder seleccionar el
# el grupo de lirios que queremos en cada momento
ini, fin = 0, 0

# Dibujamos las gráficas de cada una de las clases pintando
# los puntos con los colores indicados en el vector de colores
# y poniendole como etiqueta el nombre de la clase
for i in np.arange(clases.size):
    fin += counts[i]
    plt.scatter(X[ini:fin, 0], X[ini:fin, 1], c=colores[i], label=clases[i])
    ini += counts[i]

# Fijamos el título y las etiquetas de los ejes
plt.title('Datos de iris')
plt.xlabel(carac[0])
plt.ylabel(carac[1])
    
# Fijamos los límites de valor en ambos ejes
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Centramos la gráfica según los nuevos límites indicados
# en las fuciones anteriores
plt.xticks(())
plt.yticks(())    

# Creamos la leyenda
plt.legend()

plt.show(fig1)



####################################################
###                    Parte 2                   ###
####################################################

# Esta función divide el conjunto de los datos en training y test conservando
# la proporción de elementos de cada clase, para conseguirlo voy a realizar la
# separación de los datos de cada clase por separado, y después uniendolos todos.
# La función además de devolver los datos separados en training y test, devuelve
# los indices que pertenecen a cada clase en ambos conjuntos de datos.
def div_propor_train_test(data, indices, test_size):
    # Realizamos un conteo de los elementos de cada clase
    unique, counts = np.unique(indices, return_counts=True)
    
    ini, fin = 0, counts[0]
    
    # Realizamos la división de la primera clase
    train, test = train_test_split(data[ini:fin,:], test_size = test_size)
    indiceTrain = np.repeat(0,train[:,0].size)
    indiceTest = np.repeat(0,test[:,0].size)
    ini = fin
    
    # Realizamos la división del resto de clases
    for i in np.arange(1,counts.size):
        fin += counts[i]
        auxTrain, auxTest = train_test_split(data[ini:fin,:],
                                             test_size = test_size)
        train = np.concatenate((train, auxTrain), axis=0)
        indiceTrain = np.concatenate((indiceTrain,
                                      np.repeat(i,auxTrain[:,0].size)), axis=0)
        test = np.concatenate((test, auxTest), axis=0)
        indiceTest = np.concatenate((indiceTest,
                                     np.repeat(i,auxTest[:,0].size)), axis=0)
        ini = fin
        
    return train, indiceTrain, test, indiceTest
    

# Realizar la separación
train, indiceTrain, test, indiceTest = div_propor_train_test(X, y, 0.25)

# Mostrar información sobre la separación
unique, conteoTotal = np.unique(y, return_counts=True)
unique, conteoTrain = np.unique(indiceTrain, return_counts=True)
unique, conteoTest = np.unique(indiceTest, return_counts=True)


print('\n\n## Segunda parte ##\n')
print('Total → Tam. datos: {} | Tam. training: {} ({}%) |Tam. test: {} ({}%)'
      .format(y.size, indiceTrain.size, indiceTrain.size/y.size*100,
              indiceTest.size, indiceTest.size/y.size*100))
for i in np.arange(clases.size):
    print('Clase {} → Tam. datos: {} | Tam. training: {} ({}%) |Tam. test: {} ({}%)'
          .format(clases[i], conteoTotal[i], conteoTrain[i],
                  conteoTrain[i]/conteoTotal[i]*100, conteoTest[i],
                  conteoTest[i]/conteoTotal[i]*100))



####################################################
###                    Parte 3                   ###
####################################################

print('\n\n## Tercera parte ##')
valores = np.linspace(0, 4*np.pi, 100)

resultSin = np.sin(valores)
resultCos = np.cos(valores)
resultTanh = np.tanh(resultSin + resultCos)

# Creo la segunda figura
fig2 = plt.figure(2, figsize=(8,6))

# Creamos un vector de colores
colores = np.array(['green', 'black', 'red'])

# Dibujamos las gráficas de cada una de las operaciones
plt.plot(valores, resultSin, '--', c=colores[0], label='sin(x)')
plt.plot(valores, resultCos, '--', c=colores[1], label='cos(x)')
plt.plot(valores, resultTanh, '--', c=colores[2], label='tanh(sin(x)+cos(x))')

# Fijamos el título
plt.title('Resultados de las operaciones')
    
# Creamos la leyenda
plt.legend()

plt.show(fig2)










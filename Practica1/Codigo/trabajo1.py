# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Mario Carmona Segovia
"""

import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')

# Funcion E
def E(u,v):
    return np.float64( (u**3*np.exp(v - 2) - 2*v**2*np.exp(-u))**2 )

# Primera derivada parcial de E con respecto a u
def dEu(u,v):
    return np.float64( 2 * (u**3*np.exp(v - 2) - 2*v**2*np.exp(-u)) * 
                       (3*np.exp(v - 2)*u**2 + 2*v**2*np.exp(-u)) )
    
# Primera derivada parcial de E con respecto a v
def dEv(u,v):
    return np.float64( 2 * (u**3*np.exp(v - 2) - 2*v**2*np.exp(-u)) * 
                       (u**3*np.exp(v - 2) - 4*np.exp(-u)*v) )

# Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


# Funcion F
def F(x,y):
    return np.float64( (x + 2)**2 + 2*(y - 2)**2 + 
                       2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) )

# Primera derivada parcial de F con respecto a x
def dFx(x,y):
    return np.float64( 2*(x + 2) + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) )

# Primera derivada parcial de F con respecto a y
def dFy(x,y):
    return np.float64( 4*(y - 2) + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) )

# Gradiente de F
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])


# Función que calcula un punto con el gradiente descendente
def gradient_descent(funcion, gradienteFuncion, w_o, eta, maxIter, minError2get):
    iterations = 0
    # Se asigna el punto inicial
    w_j = w_o
    # Se calcula su error
    error = funcion(w_o[0], w_o[1])
    # Se guarda el valor del error para la posterior gráfica
    valores = [error]
    while iterations < maxIter and error > minError2get:
        # Se obtiene el nuevo punto
        w_j = w_j - eta * gradienteFuncion(w_j[0], w_j[1])
        # Se calcula su error
        error = funcion(w_j[0], w_j[1])
        iterations += 1
        # Se guarda el valor del nuevo error
        valores.append(error)
    
    return w_j, iterations, valores
     
# Función para dibujar la gráfica de una función
def dibujar_funcion(funcion, title, label_x, label_y, label_z, con_marca=False,
                    coordenadas_marca=None):
    x = np.linspace(-30, 30, 50)
    y = np.linspace(-30, 30, 50)
    X, Y = np.meshgrid(x, y)
    Z = funcion(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap='jet')
    
    # Si se quiere dibujar una marca en el punto mínimo obtenido 
    # se hará verdadera la condicción y se lanza esta parte de código
    if con_marca:
        min_point = np.array([coordenadas_marca[0],coordenadas_marca[1]])
        min_point_ = min_point[:, np.newaxis]
        ax.plot(min_point_[0], min_point_[1], funcion(min_point_[0], 
                                                      min_point_[1]), 'r*', 
                                                      markersize=10)
        
    ax.set(title=title)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)
    
    plt.show(fig)


# Función para dibujar la gráfica relación error/iteraciones de una lista
# de valores
def dibujar_rela_error_iter(valores, title, label_x, label_y):
    fig = plt.figure()
    
    plt.plot(valores, c='red')
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    
    plt.show(fig)


# Función para dibujar la gráfica relación error/iteraciones de dos listas
# de valores
def dibujar_rela_error_iter_multi(valores, title, label_x, label_y, label_legend):
    fig = plt.figure()
    
    colores_linea = ['red', 'green']
    
    for i in np.arange(len(valores)):
        plt.plot(valores[i], c=colores_linea[i], label=label_legend[i])
        
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    
    plt.legend()
    
    plt.show(fig)


###############################################################################
###############################################################################


print('Ejercicio 1.2\n')

eta = 0.1                               # Tasa de aprendizaje
maxIter = 10000000000                   # Número máximo de iteraciones 
error2get = 1e-14                       # Mínimo error
initial_point = np.array([1.0,1.0])     # Punto inicial

# Obtener un punto que minimiza el error, el número de iteraciones realizadas,
# y los valores obtenidos en cada iteración mediante gradiente descendente
w, it, valores = gradient_descent(E, gradE, initial_point, eta, maxIter, error2get)

print ('Tasa de aprendizaje: ', eta)
print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Error obtenido: ', E(w[0], w[1]))


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D

# Figura 1

dibujar_rela_error_iter(valores,'Ejercicio 1.2. Relación Error / Iteraciones',
                        'Nº Iteraciones','Error')

# Figura 2

dibujar_funcion(E,'Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente',
                'u','v','E(u,v)',True,w)


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################


print('Ejercicio 1.3\n')

print('\tApartado A\n')


eta = 0.01                              # Tasa de aprendizaje
maxIter = 50                            # Número máximo de iteraciones 
error2get = -np.Infinity                # Mínimo error
initial_point = np.array([-1.0,1.0])    # Punto inicial

# Obtener un punto que minimiza el error, el número de iteraciones realizadas,
# y los valores obtenidos en cada iteración mediante gradiente descendente
w, it, valores1 = gradient_descent(F, gradF, initial_point, eta, maxIter, error2get)

print ('Tasa de aprendizaje: ', eta)
print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Error obtenido: ', F(w[0], w[1]))


eta = 0.1                               # Tasa de aprendizaje
maxIter = 50                            # Número máximo de iteraciones 
error2get = -np.Infinity                # Mínimo error
initial_point = np.array([-1.0,1.0])    # Punto inicial

# Obtener un punto que minimiza el error, el número de iteraciones realizadas,
# y los valores obtenidos en cada iteración mediante gradiente descendente
w, it, valores2 = gradient_descent(F, gradF, initial_point, eta, maxIter, error2get)

print ('\nTasa de aprendizaje: ', eta)
print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Error obtenido: ', F(w[0], w[1]))


# Figura 3

valores = [valores1,valores2]
label_legend = ['eta = 0.01', 'eta = 0.1']

dibujar_rela_error_iter_multi(valores,'Ejercicio 1.3. Relación Error / Iteraciones',
                              'Nº Iteraciones','Error',label_legend)

# Figura 4

dibujar_funcion(F,'Ejercicio 1.3. Función sobre la que se calcula el descenso de gradiente',
                'x','y','f(x,y)')


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################


print('\tApartado B')

eta = 0.01                  # Tasa de aprendizaje
maxIter = 50                # Número máximo de iteraciones 
error2get = -np.Infinity    # Mínimo error

for x, y in [(-0.5,-0.5),(1,1),(2.1,-2.1),(-3,3),(-2,2)]:
    initial_point = np.array([x,y])     # Punto inicial
    
    # Obtener un punto que minimiza el error, el número de iteraciones realizadas,
    # y los valores obtenidos en cada iteración mediante gradiente descendente
    w, it, valores = gradient_descent(F, gradF, initial_point, eta, maxIter, error2get)
    
    print ('\nTasa de aprendizaje: ', eta)
    print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('Error obtenido: ', F(w[0], w[1]))


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
###############################################################################


print('EJERCICIO SOBRE REGRESION LINEAL\n')

print('Ejercicio 2.1\n')


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y


# Funcion para calcular el error cuadrático
def Err(x,y,w):
    error_cuadratico = (x.dot(w) - y)**2
    
    error_medio = error_cuadratico.mean()
    
    return error_medio


# Derivada del error cuadrático
def gradienteError(x,y,w):
    diferencia = x.dot(w) - y.reshape(-1,1)
    
    sumatoria = x*diferencia
    
    derivada = 2 * np.mean(sumatoria, axis=0)
    
    return derivada.reshape(-1,1)



# Gradiente Descendente Estocastico
def sgd(x, y, tam_batch, eta, maxIter):
    iterations = 0
    # Vector que contiene el valor de los índices de x
    indices = np.arange(0,x.shape[0],1)
    # Barajar los ejemplos
    np.random.shuffle(indices)
    # Pesos del modelo
    w_j = np.zeros((x.shape[1],1), np.float64)
    # Variables que indican el inicio y el fin de la parte del vector índices
    # que se va a usar para mover los pesos 
    i, j = 0, tam_batch
    while iterations < maxIter:
        # Se tiene en cuenta si hay suficientes datos como para completar 
        # otro minibatch en la siguiente iteración
        if j+tam_batch > x.shape[0]:
            j = x.shape[0]
        
        w_j = w_j - eta * gradienteError(x[indices[i:j:1]], y[indices[i:j:1]], w_j)
        
        if j == x.shape[0]:
            # Barajar los ejemplos
            np.random.shuffle(indices)
            i, j = 0, tam_batch
        else:
            i += tam_batch
            j += tam_batch
        
        iterations += 1
    
    return w_j, iterations


# Pseudoinversa	
def pseudoinverse(x, y):
    # Primero se traspone la matriz x
    x_traspuesta = x.T
    
    # Segundo se calcula el producto escalar de la traspuesta de x y x
    x_produc_esca = x_traspuesta.dot(x)
    
    # Tercero se calcula la inversa del producto escalar obtenido
    x_inversa = np.linalg.inv(x_produc_esca)
    
    # Cuarto se calcula la pseudo-inversa, como el producto escalar de
    # la inversa de x y la traspuesta de x
    x_pseudoinversa = x_inversa.dot(x_traspuesta)
    
    # Por último los pesos son resultado del producto escalar de la pseudoinversa
    # de x y la matriz y.
    w = x_pseudoinversa.dot(y)
    
    return w
    

# Función para calcular el valor de x2 dado x1 y los pesos
def obtener_punto_del_modelo(x1,w):
    return -(w[0]+x1*w[1])/w[2]


# Función para dibujar el gráfico de puntos junto con la recta del modelo
def pintar_mapa_con_func(X,Y,w,labels,colores_labels,nombre_labels,lim_x,
                           lim_y,ejes_equi,title=None,label_x=None,label_y=None):
    X = X[:,1:3:1]
    
    fig = plt.figure()
    
    for label in labels:
        # Se obtiene los indices de los elementos con la etiqueta que se
        # está iterando
        indices_label = np.where(Y == label)
        # Se pintan todos los puntos obtenidos con la etiqueta de la iteración
        plt.scatter(X[indices_label,0], X[indices_label,1], 
                    c=colores_labels[label], label=nombre_labels[label])               

    plt.plot(lim_x,obtener_punto_del_modelo(lim_x, w), c='green', label='Modelo')

    if title != None:
        plt.title(title)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
    
    plt.xlim(lim_x)
    if lim_y[0] != -np.Inf:
        plt.ylim(lim_y)
    
    if ejes_equi == True:
        # Con este método se indica que la distancia en el eje x es equivalente
        # a la distancia en el eje y.
        plt.gca().set_aspect('equal')
    
    plt.legend()

    plt.show(fig)


###############################################################################
###############################################################################


# Etiquetas para las salidas
label5 = 1
label1 = -1


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')



print ('Distintas combinaciones de nº de iteraciones y tamaño de minibatch:\n')

eta = 0.01                          # Tasa de aprendizaje
maxIterPrueba = [200,500,1000]      # Número máximo de iteraciones 
tam_batch_prueba = [50,100,150]     # Tamaño del batch
for i in np.arange(0,3,1):
    for j in np.arange(0,3,1):
        w, iteraciones = sgd(x,y,tam_batch_prueba[j],eta,maxIterPrueba[i])
        
        print ("Iteraciones = ",maxIterPrueba[i]," Tam. Batch = ",tam_batch_prueba[j],
               " Ein: ",Err(x,y,w)," Eout: ",Err(x_test, y_test, w))
    print('\n')    
    

tiempo_ini = time.time()
# Obtener los pesos del hiperplano mediante la pseudoinversa
w = pseudoinverse(x,y)
tiempo_fin = time.time()
print ('\nBondad del resultado para pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print ("Tiempo ejec.: ", tiempo_fin-tiempo_ini)


eta = 0.01         # Tasa de aprendizaje
maxIter = 500      # Número máximo de iteraciones 
tam_batch = 50     # Tamaño del batch

tiempo_ini = time.time()
# Obtener los pesos del hiperplano mediante el gradiente descendente estocástico
w, iterations = sgd(x,y,tam_batch,eta,maxIter)
tiempo_fin = time.time()
print ('\nBondad del resultado para grad. descendente estocastico:\n')
print ("Nº Iteraciones: ", iterations)
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print ("Tiempo ejec.: ", tiempo_fin-tiempo_ini)


# Conjunto de etiquetas asociadas a sus colores
labels = [label1, label5]
# Asociación de un color a cada etiqueta
colores_labels = {label1: 'blue', label5: 'red'}
# Asociación de un label a cada etiqueta
nombre_labels = {label1: 'Número → 1', label5: 'Número → 5'}

lim_x, lim_y = np.array([0,1]), np.array([-np.Inf,np.Inf])

# Figura 5

# Pintar mapa con los datos de train
pintar_mapa_con_func(x,y,w,labels,colores_labels,nombre_labels,lim_x,lim_y,False,
                     'Train','Intensidad promedio','Simetria')

# Figura 6

# Pintar mapa con los datos de test
pintar_mapa_con_func(x_test,y_test,w,labels,colores_labels,nombre_labels,lim_x,
                     lim_y,False,'Test','Intensidad promedio','Simetria')

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################


print('Ejercicio 2.2\n')


# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
    X = np.random.uniform(-size,size,(N,d))
    unos = np.ones((X.shape[0],1),dtype=np.float64)
    X1 = np.concatenate((unos,X), axis=1)
    return X1

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1 - 0.2)**2 + x2**2 - 0.6)


# Función para obtener las etiquetas de los datos X
def obtenerEtiquetas(funcion, X, labels, porcentaje):
    X = X[:,1:3:1]
    Y = np.empty((X.shape[0],1))
    for i in np.arange(0,X.shape[0],1):
        Y[i] = funcion(X[i][0], X[i][1])

    indices_a_cambiar = np.random.choice(Y.shape[0],int(Y.shape[0]*porcentaje),
                                         replace=False)
    
    # Cambiar un porcentaje de las etiquetas para simular
    # el ruido en las etiquetas
    indices_label1 = np.where(Y[indices_a_cambiar] == labels[0])
    indices_label2 = np.where(Y[indices_a_cambiar] == labels[1])
    
    Y[indices_label1[0]] = labels[1]
    Y[indices_label2[0]] = labels[0]

    return Y


# Función para pintar un mapa de puntos con los datos
def pintar_mapa(X,Y,labels,colores_labels,nombre_labels,ejes_equi):
    X = X[:,1:3:1]
    
    fig = plt.figure()
    
    for label in labels:
        # Se obtiene los indices de los elementos con la etiqueta que se
        # está iterando
        indices_label = np.where(Y == label)
        # Se pintan todos los puntos obtenido con la etiqueta de la iteración
        plt.scatter(X[indices_label,0], X[indices_label,1], 
                    c=colores_labels[label], label=nombre_labels[label])  

    if ejes_equi == True:
        # Con este método se indica que la distancia en el eje x es equivalente
        # a la distancia en el eje y.
        plt.gca().set_aspect('equal')

    plt.legend()

    plt.show(fig)


###############################################################################
###############################################################################


# Conjunto de etiquetas asociadas a sus colores
colores_labels = {label1: 'blue', label5: 'orange'}


# Generamos una muestra de entrenamiento
N = 1000                # Número de puntos que se van a crear
X = simula_unif(N,2,1)
porcentaje = 0.1        # Porcentaje de ruido


###############################################################################


print('\tApartado B\n')

Y = obtenerEtiquetas(f, X, labels, porcentaje)

pintar_mapa(X,Y,labels,colores_labels,nombre_labels,True)

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################


print('\tApartado C\n')

eta = 0.01         # Tasa de aprendizaje
maxIter = 500      # Número máximo de iteraciones 
tam_batch = 50     # Tamaño del batch 

w, iterations = sgd(X, Y, tam_batch, eta, maxIter)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Nº Iteraciones: ", iterations)
print ("Ein: ", Err(X,Y,w))


lim_x, lim_y = np.array([-1,1]), np.array([-1,1])


# Figura 7

pintar_mapa_con_func(X,Y,w,labels,colores_labels,nombre_labels,lim_x,lim_y,True)

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################


print('\tApartado D (Lineal)')


num_iteraciones = 1
Ein_total = np.empty((num_iteraciones,1),dtype=np.float64)
Eout_total = np.empty((num_iteraciones,1),dtype=np.float64)

N = 1000
porcentaje = 0.1    # Porcentaje de ruido
eta = 0.01          # Tasa de aprendizaje
maxIter = 500       # Número máximo de iteraciones 
tam_batch = 50      # Tamaño del batch

tiempo_ini = time.time()
for i in np.arange(0,num_iteraciones,1):
    # Generar muestra de train
    X = simula_unif(N,2,1)
    Y = obtenerEtiquetas(f, X, labels, porcentaje)

    # Obtener modelo
    w, iterations = sgd(X, Y, tam_batch, eta, maxIter)
    
    # Calcular error en la muestra de train
    Ein_total[i] = Err(X,Y,w)
    
    # Generar muestra de test
    X = simula_unif(N,2,1)
    Y = obtenerEtiquetas(f, X, labels, porcentaje)
    
    # Calcular error en la muestra de test
    Eout_total[i] = Err(X,Y,w)
    
tiempo_fin = time.time()

# Calcular los errores medios
Ein_medio = Ein_total.mean()
Eout_medio = Eout_total.mean()


print ('\nBondad del resultado para grad. descendente estocastico:\n')
print ("Nº Iteraciones: ", iterations)
print ("Ein medio: ", Ein_medio)
print ("Eout medio: ", Eout_medio)
print ("Tiempo ejec.: ", tiempo_fin-tiempo_ini)


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################


print('\tApartado D (No Lineal)')

# Función para convertir el vector de características lineal en un vector
# de características no lineal
def convertir_a_no_lineal(X):
    x1_x2 = X[:,1]*X[:,2]
    x1_x2 = x1_x2.reshape((-1,1))
    x1_cuadrado = X[:,1]**2
    x1_cuadrado = x1_cuadrado.reshape((-1,1))
    x2_cuadrado = X[:,2]**2
    x2_cuadrado = x2_cuadrado.reshape((-1,1))
    return np.concatenate((X, x1_x2, x1_cuadrado, x2_cuadrado), axis=1)

###############################################################################


num_iteraciones = 1000
Ein_total = np.empty((num_iteraciones,1),dtype=np.float64)
Eout_total = np.empty((num_iteraciones,1),dtype=np.float64)

N = 1000
porcentaje = 0.1    # Porcentaje de ruido
eta = 0.01          # Tasa de aprendizaje
maxIter = 500       # Número máximo de iteraciones 
tam_batch = 50      # Tamaño del batch  

tiempo_ini = time.time()
for i in np.arange(0,num_iteraciones,1):
    # Generar muestra de train
    X = simula_unif(N,2,1)
    Y = obtenerEtiquetas(f, X, labels, porcentaje)
    X = convertir_a_no_lineal(X)

    # Obtener modelo
    w, iterations = sgd(X, Y, tam_batch, eta, maxIter)
    
    # Calcular error en la muestra de train
    Ein_total[i] = Err(X,Y,w)
    
    # Generar muestra de test
    X = simula_unif(N,2,1)
    Y = obtenerEtiquetas(f, X, labels, porcentaje)
    X = convertir_a_no_lineal(X)
    
    # Calcular error en la muestra de test
    Eout_total[i] = Err(X,Y,w)
    
tiempo_fin = time.time()

# Calcular los errores medios
Ein_medio = Ein_total.mean()
Eout_medio = Eout_total.mean()


print ('\nBondad del resultado para grad. descendente estocastico:\n')
print ("Nº Iteraciones: ", iterations)
print ("Ein medio: ", Ein_medio)
print ("Eout medio: ", Eout_medio)
print ("Tiempo ejec.: ", tiempo_fin-tiempo_ini)


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
###############################################################################


print('EJERCICIO BONUS\n')

print('Ejercicio 1\n')


# Segunda derivada parcial de F con respecto a x
def d2Fx(x,y):
    return np.float64( 2 - 8*(np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) )

# Segunda derivada parcial de F con respecto a y
def d2Fy(x,y):
    return np.float64( 4 - 8*(np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) )

# Segunda derivada parcial de F con respecto a x e y (= respecto a y y x)
def d2Fxy(x,y):
    return np.float64( 8*(np.pi**2)*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) )

# Función para obtener la matriz hessiana de la función F
def obtener_matriz_hessiana(x,y):
    return np.array([[d2Fx(x,y), d2Fxy(x,y)], [d2Fxy(x,y), d2Fy(x,y)]])

# Función que calcula los pesos con el método de Newton
def method_newton(funcion, gradienteFuncion, funcMatrizHessiana, dFunc_x, 
                  dFunc_y, w_o, maxIter):
    iterations = 0
    w_j = w_o
    error = funcion(w_o[0], w_o[1])
    valores = [error]
    while iterations < maxIter:
        # Matriz hessiana de la función
        H = np.linalg.inv(funcMatrizHessiana(w_j[0], w_j[1]))
        w_j = w_j - H.dot(gradienteFuncion(w_j[0], w_j[1]))
        error = funcion(w_j[0], w_j[1])
        iterations += 1
        valores.append(error)
    
    return w_j, iterations, valores


###############################################################################
###############################################################################


print('\tApartado A\n')


eta = 0.01                              # Tasa de aprendizaje
maxIter = 50                            # Número máximo de iteraciones 
error2get = -np.Infinity                # Mínimo error
initial_point = np.array([-1.0,1.0])    # Punto inicial

w, it, valores1 = gradient_descent(F, gradF, initial_point, eta, maxIter, error2get)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ('Tasa de aprendizaje: ', eta)
print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Error obtenido: ', F(w[0], w[1]))


maxIter = 50                            # Número máximo de iteraciones 
initial_point = np.array([-1.0,1.0])    # Punto inicial

w, it, valores2 = method_newton(F, gradF, obtener_matriz_hessiana, dFx, dFy, 
                                initial_point, maxIter)

print ('\nBondad del resultado para método de Newton:\n')
print ('Tasa de aprendizaje: ', eta)
print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ('Error obtenido: ', F(w[0], w[1]))



valores = [valores1,valores2]
label_legend = ['eta = 0.01 (SGD)', 'eta = 0.01 (Met. Newton)']

# Figura 8

dibujar_rela_error_iter_multi(valores,'Ejercicio 1 (Bonus). Relación Error / Iteraciones',
                              'Nº Iteraciones','Error',label_legend)


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################


print('\tApartado B')

maxIter = 50                # Número máximo de iteraciones 
error2get = -np.Infinity    # Mínimo error

for x, y in [(-0.5,-0.5),(1,1),(2.1,-2.1),(-3,3),(-2,2)]:
    initial_point = np.array([x,y])     # Punto inicial
    
    w, it, valores1 = method_newton(F, gradF, obtener_matriz_hessiana, dFx, dFy, 
                                    initial_point, maxIter)

    print ('\nTasa de aprendizaje: ', eta)
    print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('Error obtenido: ', F(w[0], w[1]))







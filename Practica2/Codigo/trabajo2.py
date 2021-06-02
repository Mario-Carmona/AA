# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Mario Carmona Segovia
"""
import numpy as np
import matplotlib.pyplot as plt
import time

# Fijamos la semilla
np.random.seed(1)

# Función para crear una nube de puntos en un cierto rango
def simula_unif(N, dim, rango):
	return np.float64(np.random.uniform(rango[0],rango[1],(N,dim)))

# Función para crear una nube de puntos mediante una distribución gaussiana
def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

# Función para obtener los valores a y b de una recta a partir de un rango de valores
def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return np.float64(a), np.float64(b)


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

# Función para pintar una nube de puntos con los datos
def pintar_nube_puntos(X, title):
    fig = plt.figure()
    plt.scatter(X[:,0], X[:,1])
    plt.xlabel('Valor de x')
    plt.ylabel('Valor de y')
    plt.title(title)
    plt.show(fig)
    
print('EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO\n')

print('Ejercicio 1.1\n')

print('Apartado A\n')

# Creación de la nube de puntos
x = simula_unif(50, 2, [-50,50])

pintar_nube_puntos(x,'Nube de puntos usando simula_unif')

print('\nApartado B\n')

# Creación de la nube de puntos
x = simula_gaus(50, 2, np.array([5,7]))

pintar_nube_puntos(x,'Nube de puntos usando una distribución Gaussiana')

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

# Función que devuelve la etiqueta de una posición respecto de la función f
def f(x, y, a, b):
	return signo(y - a*x - b)

# Función que devuelve el valor en el eje y de cierto valor del eje x
def recta(x, a, b):
    return a*x + b

# Función para obtener todas las etiquetas de una muestra respecto de una función
def obtener_etiquetas_muestra(muestra, a, b):
    etiquetas = np.empty(muestra.shape[0])
    for i in np.arange(0,muestra.shape[0],1):
        etiquetas[i] = f(muestra[i,0], muestra[i,1], a, b)
    
    return etiquetas

# Función para añadir ruido a la clasificación, modificando ciertas etiquetas
def añadir_ruido(etiquetas, labels, porcentaje_ruido):
    etiquetas_con_ruido = etiquetas.copy()
    for i in np.arange(0,len(labels),1):
        indices_label = np.where(etiquetas == labels[i])
        
        indices_a_cambiar = np.random.choice(indices_label[0].shape[0], 
                                             int(indices_label[0].shape[0]*porcentaje_ruido), 
                                             replace=False)
        
        etiquetas_con_ruido[indices_label[0][indices_a_cambiar]] = labels[(i+1) % len(labels)]
        
    return etiquetas_con_ruido


# Función para pintar una nube de puntos junto con la recta que los clasifica
def pintar_nube_puntos_clasi(X, Y, a, b, rango, labels, color_labels, title):
    fig = plt.figure()
    
    for label in labels:
        indices_label = np.where(Y == label)
        plt.scatter(X[indices_label,0],X[indices_label,1],c=color_labels[label],label=str(label))
    
    valores_y_recta = []
    for i in rango:
        valores_y_recta.append(recta(i,a,b))
    
    plt.plot(rango, valores_y_recta, label='Recta simulada')
    
    plt.xlabel('Valor de x')
    plt.ylabel('Valor de y')
    plt.xlim(rango)
    plt.ylim(rango)
    plt.title(title)
    plt.legend()
    plt.show(fig)



print('Ejercicio 1.2\n')

print('Apartado A\n')

N = 100
dim = 2
rango = [-50,50]

# Obtengo los valores a y b de la recta simulada
a, b = simula_recta(rango)

# Generar la muestra de puntos 2D
muestra = simula_unif(N,dim,rango)

# Obtener las etiquetas de la muestra
etiquetas = obtener_etiquetas_muestra(muestra, a, b)

# Valores de las etiquetas para todos los ejercicios de la práctica
labels = [1, -1]
# Color de cada etiqueta para todos los ejercicios de la práctica
color_labels = {labels[0]: 'yellow', labels[1]: 'purple'}

# Mostrar la nube de puntos junto con la recta simulada
title = 'Gráfica 2D del etiquetado'
pintar_nube_puntos_clasi(muestra, etiquetas, a, b, rango, labels, color_labels, title)


input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

print('Apartado B\n')

# Porcentaje en tanto por uno, de etiquetas que van a ser modificadas
porcentaje_ruido = 0.1

# Se generan las etiquetas con ruido
etiquetas_con_ruido = añadir_ruido(etiquetas, labels, porcentaje_ruido)

# Mostrar la nube de puntos con ruido junto con la recta simulada
title = 'Gráfica 2D del etiquetado con un ruido del 10%'
pintar_nube_puntos_clasi(muestra, etiquetas_con_ruido, a, b, rango, labels, color_labels, title)


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta


# Función para generar la gráfica de regiones pasando el valor de a y b de la función que clasifica
def plot_datos_cuad_f(X, y, fz, a, b, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid,a,b)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T,a,b).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()

# Función para generar la gráfica de regiones sin pasar el valor de a y b de la función que clasifica
def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()


# Funciones del ejercicio

# Función simulada utilizada en los dos apartados anteriores  
def f0(x,a,b):
	return np.float64(x[:,1] - a*x[:,0] - b)

# f(x,y) = (x - 10)² + (y - 20)² - 400
def f1(x):
    return np.float64((x[:,0] - 10)**2 + (x[:,1] - 20)**2 - 400)

# f(x,y) = 0,5(x + 10)² + (y - 20)² - 400
def f2(x):
    return np.float64(0.5*(x[:,0] + 10)**2 + (x[:,1] - 20)**2 - 400)

# f(x,y) = 0,5(x - 10)² - (y + 20)² - 400
def f3(x):
    return np.float64(0.5*(x[:,0] - 10)**2 - (x[:,1] + 20)**2 - 400)

# f(x,y) = y - 20x² - 5x + 3
def f4(x):
    return np.float64(x[:,1] - 20*x[:,0]**2 - 5*x[:,0] + 3)

    
print('Apartado C\n')

# Mostrar todas las funciones con los datos generados en el apartado anterior

plot_datos_cuad_f(muestra, etiquetas_con_ruido, f0, a, b, 'Funcion f(x,y) = y - ax - b con etiquetado del 2b')

plot_datos_cuad(muestra, etiquetas_con_ruido, f1, 'Funcion f(x,y) = (x - 10)² + (y - 20)² - 400 con etiquetado del 2b')

plot_datos_cuad(muestra, etiquetas_con_ruido, f2, 'Funcion f(x,y) = 0,5(x + 10)² + (y - 20)² - 400 con etiquetado del 2b')

plot_datos_cuad(muestra, etiquetas_con_ruido, f3, 'Funcion f(x,y) = 0,5(x - 10)² - (y + 20)² - 400 con etiquetado del 2b')

plot_datos_cuad(muestra, etiquetas_con_ruido, f4, 'Funcion f(x,y) = y - 20x² - 5x + 3 con etiquetado del 2b')


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 2.1: ALGORITMO PERCEPTRON

# Función para añadir la columan de unos a la muestra de datos
def añadir_columna_de_unos(x):
    unos = np.ones((x.shape[0],1), dtype=np.float64)
    return np.concatenate((unos,x), axis=1)

# Función para obtener la etiqueta que se predice con el modelo
def obtener_predicion(w,x):
    return signo(w.T.dot(x)[0])

# Función para obtener el error dentro de la muestra
def obtener_E_in(x, y, w):
    num_eti_bien_clasi = 0
    
    for i in np.arange(0,x.shape[0],1):
        prediccion = obtener_predicion(w, x[i])
        if prediccion == y[i]:
            num_eti_bien_clasi += 1
    
    accuracy = num_eti_bien_clasi / y.shape[0]
    
    return accuracy

# Algoritmo del Perceptron
def ajusta_PLA(datos, label, max_iter, vini):
    # Inicializar los pesos
    w = vini
    
    # Inicializar las iteraciones
    iteraciones = 0
    
    mejora = True
    
    # Lista de accuracy dentro de la muestra para cada iteración del algoritmo
    lista_accuracy = []
    lista_accuracy.append(obtener_E_in(datos, label, w))
    
    while mejora and iteraciones < max_iter:
        # Se indica que por ahora no hay mejora
        mejora = False
        
        i = 0
        
        # Recorremos todos los datos comprobando si hay alguna modificación posible
        while i < datos.shape[0] and iteraciones < max_iter:
            
            # Predecimos la etiqueta del punto
            etiqueta = obtener_predicion(w,datos[i])
            
            # Si no se acierta en la predición se modifican los pesos
            if etiqueta != label[i]:
                # Se actualizan los pesos
                w = w + (label[i] * datos[i]).reshape(-1,1)
                # Se indica que se han mejorado los pesos
                mejora = True
                
            lista_accuracy.append(obtener_E_in(datos, label, w))
                
            iteraciones += 1
            i += 1
    
    return w, iteraciones, lista_accuracy

# Función para obtener el valor en el eje y para cierto valor en el eje x
def obtener_punto_del_modelo(x,w):
    return -(w[0]+x*w[1])/w[2]

# Función para pintar la nube de puntos con dos rectas
def pintar_nube_puntos_clasi_dos_rectas(X, Y, a, b, w, rango, labels, color_labels, title):
    fig = plt.figure()
    
    for label in labels:
        indices_label = np.where(Y == label)
        plt.scatter(X[indices_label,0],X[indices_label,1],c=color_labels[label],label=str(label))
    
    valores_y_recta_simu = []
    for i in rango:
        valores_y_recta_simu.append(recta(i,a,b))
        
    valores_y_recta_apren = []
    for i in rango:
        valores_y_recta_apren.append(obtener_punto_del_modelo(i,w))
    
    plt.plot(rango, valores_y_recta_simu, label='Recta simulada')
    plt.plot(rango, valores_y_recta_apren, label='Recta aprendida')
    
    plt.xlabel('Valor de x')
    plt.ylabel('Valor de y')
    plt.xlim(rango)
    plt.ylim(rango)
    plt.title(title)
    plt.legend()
    plt.show(fig)




print('EJERCICIO SOBRE MODELOS LINEALES\n')

print('Ejercicio 2.1\n')

print('Apartado A.1\n')

# Se copian los datos del ejercicio anterior
datos = muestra.copy()
datos = añadir_columna_de_unos(datos)

# Inicialización con el vector a cero
vini = np.zeros((datos.shape[1],1))

# Máximo de iteraciones
max_iter = 10000

# Obtener los pesos y las iteraciones
w, iteraciones, ignorar = ajusta_PLA(datos,etiquetas,max_iter,vini)

# Mostrar la gráfica con la nube de datos y la recta del modelo
title = 'Comparación de rectas sin ruido en el aprendizaje'
pintar_nube_puntos_clasi_dos_rectas(muestra, etiquetas, a, b, w, rango, labels, color_labels, title)

# Random initializations
iterations = np.empty((10,1))
for i in range(0,10):
    vini = np.random.rand(datos.shape[1],1)
    
    w, iteraciones, ignorar = ajusta_PLA(datos,etiquetas,max_iter,vini)
    
    iterations[i] = iteraciones
    
    
print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(iterations)))

input("\n--- Pulsar tecla para continuar ---\n")

print('Apartado A.2\n')

# Ahora con los datos del ejercicio 1.2.b

# Se copian los datos del ejercicio anterior
datos = muestra.copy()
datos = añadir_columna_de_unos(datos)

# Inicialización con el vector a cero
vini = np.zeros((datos.shape[1],1))

# Máximo de iteraciones
max_iter = 10000

# Obtener los pesos y las iteraciones
w, iteraciones, ignorar = ajusta_PLA(datos,etiquetas_con_ruido,max_iter,vini)

# Mostrar la gráfica con la nube de datos con ruido y la recta del modelo
title = 'Comparación de rectas con ruido en el aprendizaje'
pintar_nube_puntos_clasi_dos_rectas(muestra, etiquetas_con_ruido, a, b, w, rango, labels, color_labels, title)

# Random initializations
iterations = np.empty((10,1))
for i in range(0,10):
    vini = np.random.rand(datos.shape[1],1)
    
    w, iteraciones, ignorar = ajusta_PLA(datos,etiquetas_con_ruido,max_iter,vini)
    
    iterations[i] = iteraciones


print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(iterations)))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

# Función para obtener el gradiente en cierto punto
# Esta fórmula se ha simplificado porque se ha puesto como
# tamaño de mini-batch igual a 1
def gradienteError(x,y,w):
    numerador = y * x
    denominador = 1 + np.exp(y * x.dot(w))
    
    return -(numerador / denominador)

# Algoritmo de la regresión logística
def sgdRL(x, y, eta, min_error, tam_batch):
    # Inicializar el vector de pesos a cero
    w = np.zeros((x.shape[1],1))
    
    converge = False
    epocas = 0
    
    indices = np.arange(0,x.shape[0],1)
    
    # Barajamos los datos
    np.random.shuffle(indices)
    
    i, j = 0, tam_batch
    
    w_anterior = w.copy()
    
    while not converge:
        if j+tam_batch > x.shape[0]:
            j = x.shape[0]
        
        # Actualizar los pesos
        w = w - eta * gradienteError(x[indices[i:j:1]], y[indices[i:j:1]], w).T
        
        # Si ya se han recorrido todos los elementos de la muestra
        if j == x.shape[0]:
            # Barajamos los datos
            np.random.shuffle(indices)
            i, j = 0, tam_batch
            
            # Incrementar las épocas
            epocas += 1
            
            # Calcular la distancia de error entre modelos
            distancia = np.linalg.norm(w_anterior - w)
            
            # Actualizar w_anterior
            w_anterior = w.copy()
            
            # Si la distancia de error es menor que el mínimo permitido
            # se indica que ha convergido el modelo
            if distancia < min_error:
                converge = True
        else:
            i += tam_batch
            j += tam_batch

    return w, epocas

# Función para obtener los valores a y b de una recta a partir de dos puntos de la muestra
def simula_recta_2(muestra):
    # Elegimos dos puntos al azar
    points = np.random.choice(muestra.shape[0], 2, replace=False)
    x1 = muestra[points[0]][0]
    x2 = muestra[points[1]][0]
    y1 = muestra[points[0]][1]
    y2 = muestra[points[1]][1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return np.float64(a), np.float64(b)

# Función para obtener el error fuera de la muestra
def obtener_E_out(test, etiquetas_test, w):
    num_eti_bien_clasi = 0
    
    for i in np.arange(0,test.shape[0],1):
        prediccion = obtener_predicion(w, test[i])
        if prediccion == etiquetas_test[i]:
            num_eti_bien_clasi += 1
    
    accuracy = num_eti_bien_clasi / etiquetas_test.shape[0]
    
    return accuracy


print('Apartado B\n')


rango = [0, 2]

dim = 2

# En la explicación de la práctica se indica
# que es recomendable que el tamaño del batch valga 1
tam_batch = 1

min_error = 0.01

eta = 0.01

# Generar los datos de la muestra
muestra = simula_unif(N, dim, rango)
muestra = añadir_columna_de_unos(muestra)

# Creamos una recta con dos puntos al azar de la muestra
a, b = simula_recta_2(muestra[:,1:])

# Obtenemos las etiquetas de la muestra
etiquetas = obtener_etiquetas_muestra(muestra[:,1:], a, b)
 
   
# Ejemplo de recta obtenida con el aprendizaje

# Obtener los pesos y las épocas usadas
w, epocas = sgdRL(muestra, etiquetas, eta, min_error, tam_batch)

# Mostrar la nube de datos junto con la recta simulada y el modelo aprendido
title = 'Nube de puntos con el modelo ajustado'
pintar_nube_puntos_clasi_dos_rectas(muestra[:,1:], etiquetas, a, b, w, rango, labels, color_labels, title)

input("\n--- Pulsar tecla para continuar ---\n")

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


num_repeticiones = 100

Accuracy = np.empty((num_repeticiones,1))
num_epocas = np.empty((num_repeticiones,1))

for i in np.arange(0,num_repeticiones,1):
    
    # Generar la muestra
    muestra = simula_unif(N, dim, rango)
    muestra = añadir_columna_de_unos(muestra)
    
    # Obtener las etiquetas
    etiquetas = obtener_etiquetas_muestra(muestra[:,1:], a, b)
    
    # Obtener los pesos y las épocas usadas
    w, epocas = sgdRL(muestra, etiquetas, eta, min_error, tam_batch)
    
    # Generar los datos de test
    test = simula_unif(1000, 2, rango)
    test = añadir_columna_de_unos(test)
    
    # Obtener las etiquetas del test
    etiquetas_test = obtener_etiquetas_muestra(test[:,1:], a, b)
    
    # Obtener el error fuera de la muestra o del test
    # Se obtiene tanto el número de etiquetas bien clasificadas (accuracy), 
    # como el riego empírico (ERM)
    accuracy = obtener_E_out(test, etiquetas_test, w)
    
    Accuracy[i] = accuracy
    num_epocas[i] = epocas
    

print('\nValor medio de Accuracy obtenido con N = 100: {} %'.format(np.mean(Accuracy) * 100))
print('\nValor medio de epocas necesario para converger: {}'.format(np.mean(num_epocas)))


input("\n--- Pulsar tecla para continuar ---\n")


# Ejemplo de los datos de test con el modelo ajustado.

print('\nEjemplo de los datos de test junto con la recta aprendida')

title = 'Datos de test con el modelo ajustado'
pintar_nube_puntos_clasi_dos_rectas(test[:,1:], etiquetas_test, a, b, w, rango, labels, color_labels, title)


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Función para mostrar una nube de puntos
def mostrar_datos(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
    ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
    ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title=title)
    ax.set_xlim((0, 1))
    ax.set_ylim([-7,0])
    plt.legend()
    plt.show()

# Función para mostrar una nube de puntos con la recta aprendida
def mostrar_datos_con_recta(x, y, w, rango, title):
    fig, ax = plt.subplots()
    ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
    ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
    
    valores_y = []
    for i in rango:
        valores_y.append(obtener_punto_del_modelo(i,w))
    
    ax.plot(rango, valores_y, label='Recta aprendida')
    ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title=title)
    ax.set_xlim(rango)
    ax.set_ylim([-7,0])
    plt.legend()
    plt.show()


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])

print('\nDatos de training y test del problema')
      
# Mostramos los datos de training
title = 'Digitos Manuscritos (TRAINING)'
mostrar_datos(x, y, title)

# Mostramos los datos de test
title = 'Digitos Manuscritos (TEST)'
mostrar_datos(x_test, y_test, title)


input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

# Funcion para calcular el error cuadrático
def Err(x,y,w):
    error_cuadratico = (x.dot(w) - y)**2
    
    error_medio = error_cuadratico.mean()
    
    return error_medio


# Derivada del error cuadrático
def gradienteError_2(x,y,w):
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
        
        w_j = w_j - eta * gradienteError_2(x[indices[i:j:1]], y[indices[i:j:1]], w_j)
        
        if j == x.shape[0]:
            # Barajar los ejemplos
            np.random.shuffle(indices)
            i, j = 0, tam_batch
        else:
            i += tam_batch
            j += tam_batch
        
        iterations += 1
    
    return w_j, iterations

print ("\nAlgoritmo SGD\n")

# Asignar tasa de aprendizaje
eta = 0.01

# Asignar tamaño del mini-batch
tam_batch = 50

# Asignar el número máximo de iteraciones a realizar
maxIter = 10000

# Obtener los pesos y las iteraciones usadas
tiempo_ini = time.time()
w, iteraciones = sgd(x, y, tam_batch, eta, maxIter)
tiempo_fin = time.time()
tiempo_sgd = tiempo_fin-tiempo_ini

# Mostramos el tiempo obtenido
print ("Tiempo ejec. SGD: ", tiempo_sgd)

# Fijar rango de valores del eje x
rango = [0,1]

# Mostramos los datos de training
title = 'Digitos Manuscritos (TRAINING) con SGD'
mostrar_datos_con_recta(x, y, w, rango, title)

# Mostrar error dentro de la muestra
accuracy = obtener_E_in(x, y, w)
print('\nValor de Accuracy obtenido dentro de la muestra: {} %'.format(accuracy * 100))


# Mostramos los datos de test
title = 'Digitos Manuscritos (TEST) con SGD'
mostrar_datos_con_recta(x_test, y_test, w, rango, title)

# Mostrar error fuera de la muestra
accuracy = obtener_E_out(x_test, y_test, w)
print('\nValor de Accuracy obtenido fuera de la muestra: {} %'.format(accuracy * 100))


# Se copian los pesos obtenido para usarlo en la variante del Pocket, 
# que toma como pesos iniciales los pesos obtenidos con SGD
w_sgd = w.copy()


input("\n--- Pulsar tecla para continuar ---\n")


#POCKET ALGORITHM

# Algoritmo Pocket
def pocket(x, y, max_iter, vini):
    
    # Inicializar pesos
    w = vini.copy()
    
    # Obtener los mejores pesos por el momento y su accuracy
    w_mejor = w.copy()
    mejor_accuracy = obtener_E_in(x, y, w_mejor)
    
    mejora = True
    
    iteraciones = 0
    
    # Lista de accuracy dentro de la muestra para cada iteración del algoritmo
    lista_accuracy = []
    lista_accuracy.append(obtener_E_in(x, y, w))
    
    while mejora and iteraciones < max_iter:
        # Se indica que por ahora no hay mejora
        mejora = False
        
        i = 0
            
        # Recorremos todos los datos comprobando si hay alguna modificación posible
        while i < x.shape[0] and iteraciones < max_iter:
            
            # Predecimos la etiqueta del punto
            etiqueta = obtener_predicion(w,x[i])
            
            # Si no se acierta en la predición se modifican los pesos
            if etiqueta != y[i]:
                # Se actualizan los pesos
                w = w + (y[i] * x[i]).reshape(-1,1)
                # Se indica que se han mejorado los pesos
                mejora = True
                
                # Se comprueba si los nuevos pesos son mejores que los considerados mejores
                # por el momento
                accuracy = obtener_E_in(x, y, w)
                if accuracy > mejor_accuracy:
                    w_mejor = w.copy()
                    mejor_accuracy = accuracy
                
            lista_accuracy.append(obtener_E_in(x, y, w_mejor))
                
            iteraciones += 1
            i += 1
        
    
    return w_mejor, lista_accuracy


print ("\nAlgoritmo Pocket\n")

# Asignar el número máximo de iteraciones
max_iter = 10000

# Inicializar los pesos a cero
w = np.zeros((x.shape[1],1))

# Obtener los pesos
tiempo_ini = time.time()
w, ignorar = pocket(x, y, max_iter, w.copy())
tiempo_fin = time.time()
tiempo_pocket = tiempo_fin-tiempo_ini

# Mostramos el tiempo obtenido
print ("Tiempo ejec. Pocket: ", tiempo_pocket)

# Fijar rango de valores del eje x
rango = [0,1]

# Mostramos los datos de training
title = 'Digitos Manuscritos (TRAINING) con Pocket'
mostrar_datos_con_recta(x, y, w, rango, title)

# Mostrar error dentro de la muestra
accuracy = obtener_E_in(x, y, w)
print('\nValor de Accuracy obtenido dentro de la muestra: {} %'.format(accuracy * 100))

# Mostramos los datos de test
title = 'Digitos Manuscritos (TEST) con Pocket'
mostrar_datos_con_recta(x_test, y_test, w, rango, title)

# Mostrar error fuera de la muestra
accuracy = obtener_E_out(x_test, y_test, w)
print('\nValor de Accuracy obtenido fuera de la muestra: {} %'.format(accuracy * 100))


input("\n--- Pulsar tecla para continuar ---\n")


#SGD + POCKET ALGORITHM
  
print ("\nAlgoritmo SGD + Pocket\n")

# Obtener los pesos
tiempo_ini = time.time()
w, ignorar = pocket(x, y, max_iter, w_sgd)
tiempo_fin = time.time()
tiempo_sgd_pocket = tiempo_fin-tiempo_ini

# Mostramos el tiempo obtenido
print ("Tiempo ejec. SGD + Pocket: ", tiempo_sgd_pocket)

# Fijar rango de valores del eje x
rango = [0,1]

# Mostramos los datos de training
title = 'Digitos Manuscritos (TRAINING) con SGD + Pocket'
mostrar_datos_con_recta(x, y, w, rango, title)

# Mostrar error dentro de la muestra
accuracy = obtener_E_in(x, y, w)
print('\nValor de Accuracy obtenido dentro de la muestra: {} %'.format(accuracy * 100))

# Mostramos los datos de test
title = 'Digitos Manuscritos (TEST) con SGD + Pocket'
mostrar_datos_con_recta(x_test, y_test, w, rango, title)

# Mostrar error fuera de la muestra
accuracy = obtener_E_out(x_test, y_test, w)
print('\nValor de Accuracy obtenido fuera de la muestra: {} %'.format(accuracy * 100))


input("\n--- Pulsar tecla para continuar ---\n")

# Comparación entre PLA y Pocket

def mostrar_comparacion(lista_accuracy_1, lista_accuracy_2, label_recta_1, label_recta_2, title):
    fig = plt.figure()
    
    plt.plot(np.arange(0,len(lista_accuracy_1),1), lista_accuracy_1, label=label_recta_1, color="blue")
    
    plt.xlabel('Num. Iteraciones')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show(fig)
    
    fig = plt.figure()
    plt.plot(np.arange(0,len(lista_accuracy_2),1), lista_accuracy_2, label=label_recta_2, color="orange")
    
    plt.xlabel('Num. Iteraciones')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show(fig)
    

print ("\nComparación de la evolución del accuracy entre PLA y Pocket\n")

max_iter = 5000

w = np.zeros((x.shape[1],1))
w, ignorar, lista_accuracy_PLA = ajusta_PLA(x, y, max_iter, w)

w = np.zeros((x.shape[1],1))
w, lista_accuracy_pocket = pocket(x, y, max_iter, w)

mostrar_comparacion(lista_accuracy_PLA, lista_accuracy_pocket, 'PLA', 'Pocket',
                    'Comparación entre PLA y Pocket')


input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

print('\nNo he realizado este apartado')


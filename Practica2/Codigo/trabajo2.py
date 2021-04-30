# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Mario Carmona Segovia
"""
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.float64(np.random.uniform(rango[0],rango[1],(N,dim)))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


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

x = simula_unif(50, 2, [-50,50])

pintar_nube_puntos(x,'Nube de puntos usando simula_unif')

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

def f(x, y, a, b):
	return signo(y - a*x - b)

def recta(x, a, b):
    return a*x + b

def obtener_etiquetas_muestra(muestra, a, b):
    etiquetas = np.empty(muestra.shape[0])
    for i in np.arange(0,muestra.shape[0],1):
        etiquetas[i] = f(muestra[i,0], muestra[i,1], a, b)
    
    return etiquetas

def añadir_ruido(etiquetas, labels, porcentaje_ruido):
    etiquetas_con_ruido = etiquetas.copy()
    for i in np.arange(0,len(labels),1):
        indices_label = np.where(etiquetas == labels[i])
        
        indices_a_cambiar = np.random.choice(indices_label[0].shape[0], 
                                             int(indices_label[0].shape[0]*porcentaje_ruido), 
                                             replace=False)
        
        etiquetas_con_ruido[indices_label[0][indices_a_cambiar]] = labels[(i+1) % len(labels)]
        
    return etiquetas_con_ruido


# Función para pintar una nube de puntos con los datos
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

#CODIGO DEL ESTUDIANTE

# Generar la muestra de puntos 2D
N = 100
dim = 2
rango = [-50,50]

# Obtengo los valores a y b de la recta simulada
a, b = simula_recta(rango)

muestra = simula_unif(N,dim,rango)

etiquetas = obtener_etiquetas_muestra(muestra, a, b)

labels = [1, -1]
color_labels = {labels[0]: 'yellow', labels[1]: 'purple'}

# Dibujar la nube de puntos
title = 'Gráfica 2D del etiquetado'
pintar_nube_puntos_clasi(muestra, etiquetas, a, b, rango, labels, color_labels, title)

#

input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE

porcentaje_ruido = 0.1

etiquetas_con_ruido = añadir_ruido(etiquetas, labels, porcentaje_ruido)

title = 'Gráfica 2D del etiquetado con un ruido del 10%'
pintar_nube_puntos_clasi(muestra, etiquetas_con_ruido, a, b, rango, labels, color_labels, title)

#

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

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

def f0(x,a,b):
	return np.float64(x[:,1] - a*x[:,0] - b)

def f1(x):
    return np.float64((x[:,0] - 10)**2 + (x[:,1] - 20)**2 - 400)

def f2(x):
    return np.float64(0.5*(x[:,0] + 10)**2 + (x[:,1] - 20)**2 - 400)

def f3(x):
    return np.float64(0.5*(x[:,0] - 10)**2 - (x[:,1] + 20)**2 - 400)

def f4(x):
    return np.float64(x[:,1] - 20*x[:,0]**2 - 5*x[:,0] + 3)

    
print('Ejercicio 1.3\n')

#CODIGO DEL ESTUDIANTE

plot_datos_cuad_f(muestra, etiquetas_con_ruido, f0, a, b, 'Funcion f con etiquetado del 2b')

plot_datos_cuad(muestra, etiquetas_con_ruido, f1, 'Funcion f1 con etiquetado del 2b')

plot_datos_cuad(muestra, etiquetas_con_ruido, f2, 'Funcion f2 con etiquetado del 2b')

plot_datos_cuad(muestra, etiquetas_con_ruido, f3, 'Funcion f3 con etiquetado del 2b')

plot_datos_cuad(muestra, etiquetas_con_ruido, f4, 'Funcion f4 con etiquetado del 2b')

#

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def añadir_columna_de_unos(x):
    unos = np.ones((x.shape[0],1), dtype=np.float64)
    return np.concatenate((unos,x), axis=1)

def obtener_predicion(w,x):
    return signo(w.T.dot(x)[0])

def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
    
    # Inicializar los pesos
    w = vini
    iteraciones = 0
    mejora = True
    
    while mejora and iteraciones < max_iter:
        # Se indica que por ahora no hay mejora
        mejora = False
        
        # Recorremos todos los datos comprobando si hay alguna modificación posible
        for i in np.arange(0,datos.shape[0],1):
            
            # Predecimos la etiqueta del punto
            etiqueta = obtener_predicion(w,datos[i])
            
            # Si no se acierta en la predición se modifican los pesos
            if etiqueta != label[i]:
                w = w + (label[i] * datos[i]).reshape(-1,1)
                # Se indica que se han mejora los pesos
                mejora = True
                
            iteraciones += 1
    #
    
    return w, iteraciones  

def obtener_punto_del_modelo(x,w):
    return -(w[0]+x*w[1])/w[2]

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

#CODIGO DEL ESTUDIANTE

datos = muestra.copy()
datos = añadir_columna_de_unos(datos)

# Inicialización con le vector a cero
vini = np.zeros((datos.shape[1],1))

# Máximo de iteraciones
max_iter = 10000

w, iteraciones = ajusta_PLA(datos,etiquetas,max_iter,vini)

title = 'Comparación de recta sin ruido en el aprendizaje'
pintar_nube_puntos_clasi_dos_rectas(muestra, etiquetas, a, b, w, rango, labels, color_labels, title)

# Random initializations
iterations = []
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    
    vini = np.random.rand(datos.shape[1],1)
    
    w, iteraciones = ajusta_PLA(datos,etiquetas,max_iter,vini)
    
    iterations.append(iteraciones)
    
    #   
#
    
print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE

datos = muestra.copy()
datos = añadir_columna_de_unos(datos)

# Inicialización con le vector a cero
vini = np.zeros((datos.shape[1],1))

# Máximo de iteraciones
max_iter = 10000

w, iteraciones = ajusta_PLA(datos,etiquetas_con_ruido,max_iter,vini)

title = 'Comparación de recta con ruido en el aprendizaje'
pintar_nube_puntos_clasi_dos_rectas(muestra, etiquetas_con_ruido, a, b, w, rango, labels, color_labels, title)

# Random initializations
iterations = []
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    
    vini = np.random.rand(datos.shape[1],1)
    
    w, iteraciones = ajusta_PLA(datos,etiquetas_con_ruido,max_iter,vini)
    
    iterations.append(iteraciones)
    
    #
#

print('\nValor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def gradienteError(x,y,w):
    numerador = y * x
    denominador = 1 + np.exp(y * x.dot(w))
    
    return -(numerador / denominador)

def sgdRL(x, y, eta, min_error, tam_batch):
    #CODIGO DEL ESTUDIANTE
    
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
        
        w = w - eta * gradienteError(x[indices[i:j:1]], y[indices[i:j:1]], w).T
        
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
        
    #

    return w, epocas

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

def obtener_E_out(test, etiquetas_test, w):
    # Ver página 52 de teoría para ver fórmula
    error = np.empty((test.shape[0],1))
    
    num_eti_bien_clasi = 0
    
    for i in np.arange(0,error.shape[0],1):
        error[i] = np.log(1 + np.exp(-etiquetas_test[i] * test[i].dot(w)))
        
        prediccion = obtener_predicion(w, test[i])
        if prediccion == etiquetas_test[i]:
            num_eti_bien_clasi += 1
        
    error = np.mean(error)    
    
    accuracy = num_eti_bien_clasi / etiquetas_test.shape[0]
    
    return error, accuracy


#CODIGO DEL ESTUDIANTE

rango = [0, 2]

dim = 2

# En la explicación de la práctica se indica
# que es recomendable que el tamaño del batch valga 1
tam_batch = 1

min_error = 0.01

eta = 0.01

muestra = simula_unif(N, dim, rango)
muestra = añadir_columna_de_unos(muestra)

# Creamos una recta con dos puntos al azar de la muestra
a, b = simula_recta_2(muestra[:,1:])

etiquetas = obtener_etiquetas_muestra(muestra[:,1:], a, b)

title = 'Nube de puntos con la recta inicial'
pintar_nube_puntos_clasi(muestra[:,1:], etiquetas, a, b, rango, labels, color_labels, title)

#

input("\n--- Pulsar tecla para continuar ---\n")
    
# Ejemplo de recta obtenida con el aprendizaje

w, epocas = sgdRL(muestra, etiquetas, eta, min_error, tam_batch)

title = 'Nube de puntos con el modelo ajustado'
pintar_nube_puntos_clasi_dos_rectas(muestra[:,1:], etiquetas, a, b, w, rango, labels, color_labels, title)

input("\n--- Pulsar tecla para continuar ---\n")

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE

num_repeticiones = 100

E_out = []
Accuracy = []
num_epocas = []

for i in np.arange(0,num_repeticiones,1):
    
    muestra = simula_unif(N, dim, rango)
    muestra = añadir_columna_de_unos(muestra)
    
    etiquetas = obtener_etiquetas_muestra(muestra[:,1:], a, b)
    
    w, epocas = sgdRL(muestra, etiquetas, eta, min_error, tam_batch)
    
    test = simula_unif(1000, 2, rango)
    test = añadir_columna_de_unos(test)
    etiquetas_test = obtener_etiquetas_muestra(test[:,1:], a, b)
    
    error_out, accuracy = obtener_E_out(test, etiquetas_test, w)
    
    E_out.append(error_out)
    Accuracy.append(accuracy)
    num_epocas.append(epocas)
    

print('\nValor medio de E_out obtenido con N = 100: {}'.format(np.mean(np.asarray(E_out))))
print('\nValor medio de Accuracy obtenido con N = 100: {} %'.format(np.mean(np.asarray(Accuracy)) * 100))
print('\nValor medio de epocas necesario para converger: {}'.format(np.mean(np.asarray(num_epocas))))

#


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################

'''
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

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
'''

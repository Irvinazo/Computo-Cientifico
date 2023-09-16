import numpy as np
import matplotlib.pyplot as plt #Librería para implementar las gráficas
import time #Librería para medir tiempos
from scipy.stats import uniform as unif #importamos como unif por comodidad

#Algoritmo LUP
def LUPfact(A):
    U=A.astype(np.float64)
    L=np.identity(len(A))
    P=np.identity(len(A))

    for k in range(len(A)):
        m=max(np.abs(U[k:,k]))
        pm=np.where(np.abs(U[k:,k])==m)
        pm=pm[0][0]+k

        L[pm][0:k], L[k][0:k]=L[k][0:k].copy(), L[pm][0:k].copy()
        U[pm][k:], U[k][k:]=U[k][k:].copy(), U[pm][k:].copy()
        P[pm], P[k]=P[k].copy(), P[pm].copy()
        for i in range(k+1,len(A)):
            L[i][k]=U[i][k]/U[k][k]         #se le asigna a L el valor que anulará el valor de la entrada
            U[i][k:len(A)]=U[i][k:len(A)]-L[i][k]*U[k][k:len(A)]

    return[P,A,L,U]
#Algoritmo Cholesky
def CholFact(A):
    R=A.astype(np.float64)
    for k in range(len(A)):
        for j in range(k+1,len(A)):
            R[j,j:]=R[j,j:]-R[k,j:]*R[k,j]/R[k,k]
        R[k,k:]=R[k,k:]/np.sqrt(R[k,k])
    return(R)

#Generamos una matriz aleatoria de tamaño 300, e implementamos ambos algoritmos
n=300
np.random.seed(5) #colocamos nuestra semilla 
times_LUP=np.zeros(n,dtype=float)
times_cho=np.zeros(n,dtype=float).copy() #creamos dos vectores vacíos para guardar los 300 tiempos de ejecución
for i in range(1,n):
    M=unif.rvs(size=i**2)
    M=np.reshape(M,(i,i)) #a cada uno de los pasos i, creamos muestras aleatorias uniformes de tamaño i^2, para poder guardarlos en una matriz cuadrada de tamaño i
    A=M.T@M #Volvemos a la matriz hermitiana definida positiva
    #Registramos el primer tiempo de LUP
    T0L = time.time()
    LUPfact(A)
    TfL = time.time()
    #Registramos el último tiempo de LUP
    #Registramos el primer tiempo de Cholesky
    T0C = time.time()
    CholFact(A)
    TfC = time.time()
    #Registramos el último tiempo de Cholesky
    #Guardamos los tiempos de ejecución
    times_LUP[i]=TfL-T0L
    times_cho[i]=TfC-T0C  

#Como mera curiosidad, imprimimos los tiempos de ejecución
times_LUP
times_cho
#Creamos un vector con las entradas entre 1 y 300 para graficar los tiempos
m=np.arange(1,n+1)
m

#Graficamos los tiempos colocando una rejilla, etiquetas y títulos.
plt.plot(m,times_LUP, label="Algoritmo LUP")
plt.plot(m,times_cho, label= "Algoritmo Cholesky")
plt.grid()
plt.xlabel('Cantidad de matrices')
plt.ylabel('Tiempo')
plt.title('Comparación del tiempo de ejecución de los dos algoritmos')
plt.legend()
plt.show()


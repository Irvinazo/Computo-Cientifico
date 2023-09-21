#Importamos las librerías necesarias:

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#Cargamos la función para la descomposición QR con Gram modificado:

def GMmod(A):
    v=A.copy()
    v=v.astype(np.float64)
    R=np.zeros((np.shape(A)[1],np.shape(A)[1]))
    Q=np.zeros(np.shape(A))

    for i in range(len(R)): #Para cada una de las columnas de A:
        R[i,i]=np.sqrt((v[:,i]).T@v[:,i]) #extraemos la norma de esos vectores
        Q[:,i]=v[:,i]/R[i,i] # Y a las columnas de Q, se les añade el vector v_i normalizado
        for j in range(i+1,len(R)): #para la columna i fija, a partir del renglón entre i hasta el último renglón,
            R[i,j]=(Q[:,i]).T@v[:,j] #Rij lo rellenamos
            v[:,j]=v[:,j]-R[i,j]*Q[:,i] #actualizamos v

    return[A,Q,R]

#Cargamos back substitution

def backsub(M,a):
    
    A=M.astype(np.float64)
    b=a.astype(np.float64)
    x=np.empty(len(A),dtype=float)

    for k in range(len(A)):
        v=np.empty(k,dtype=float)
        v[0:k]=A[len(A)-1-k][len(A)-1-k+1:len(A)]*x[len(A)-1-k+1:len(A)]
        x[len(A)-1-k]=(b[len(A)-1-k]-sum(v))/A[len(A)-1-k][len(A)-1-k]

    return(x)

#Cargamos la 'solución' de sistemas vía QR

def QRsol(A,b):
    Q=GMmod(A)[1]
    R=GMmod(A)[2]
    Qb=Q.T@b
    x=backsub(R,Qb)
    return(x)


"""
Función para crear una matriz de Vandermonde de tamaño p+1 con un vector v de m datos.
Esta matriz nos devuelve, para un vector que le sea ingresado (pensemos en un vector de datos de tamaño m),
una matriz donde los renglones son las potencias de cada uno de esos datos, desde la potencia 0 hasta la p.
Es importante señalar que esta matriz recrea polinomios de grado $p$, por lo que la matriz tiene p+1 columnas
"""
def MVanderm(v,p):
    if p>len(v)-1:
        print('Error: la regresión polinomial tiene que ser con p<=m-1')    #Si se ingresa un grado de tamaño más grande que
                                                                            #el tamaño de nuestro vector de datos, arroja error.
    else:
        X=np.zeros((len(v),p+1))                                            #Creamos la matriz de tamaño adecuado.
        X[:,0]=1                                                            #La primera columna es de solo 1's.
        for i in range(1,p+1):
            X[:,i]=v**i                                                     #elevamos los datos a la potencia adecuada.
    return(X)


np.random.seed(10)   #Colocamos semillas para poder replicar los resultados de los experimentos aleatorios

"""Creamos ahora una función que nos otorgue los datos Y=sen(x_i)+e_i y los regresores X como se especifica en el ejercicio 3."""

def Regpol(m,p):
        
    e=norm.rvs(loc=0,scale=0.11,size=m) #Creamos m muestras normales con las características indicadas
    r=np.zeros(m)
    y=np.zeros(m)               #Creamos los vectores auxiliares
    v=np.zeros(m)
    for i in range(m):
        r[i]=4*np.pi*(i+1)/m
        y[i]=np.sin(r[i])+e[i]  #'r' será el vector que guarde la partición del intervalo [0,4pi] (los regresores) y 'y' serán los datos observados
        v[i]=np.sin(r[i])       #'v' será el vector que guarde los datos de los datos reales (sin ruido)
    X=MVanderm(r,p)             #Cargamos la matriz de Vandermonde con los regresores y el grado deseado
    b=QRsol(X,y)                #Hallamos la solución del sistema y=Xb. Son los coeficientes del polinomio que ajusta.
    return[y,X,b,r,v]

"""Creamos una función que nos calcule los polinomios que nos surgirán. Esto nos permitirá graficar"""

a=np.linspace(0,4*np.pi,num=50) #Creamos una rejilla de números en (0,4pi)
def pol(p,b,v):
    z=np.zeros(p+1)
    for i in range(len(a)):
        for j in range(p+1):
            z[j]=b[j]*(a[i]**j)
            v[i]=sum(z)
    return(v)

#Ejemplo 1: ajustamos un polinomio para p=3,m=100
m=100
p=2   #Especificamos los tamaños que se pueden buscar (en la tarea es de grado p-1)
pol1=np.zeros(50) #creamos un vector para guardar el polinomio ajustado

#Graficamos
plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=100,p=3')
plt.legend()
plt.show()

#Ejemplo 2: ajustamos un polinomio para p=4,m=100
m=100
p=3   
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=100,p=4')
plt.legend()
plt.show()

#Ejemplo 3: ajustamos un polinomio para p=6,m=100
m=100
p=5   
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=100,p=6')
plt.legend()
plt.show()

#Ejemplo 4: ajustamos un polinomio para p=100,m=100
m=100
p=99   
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=100,p=100')
plt.legend()
plt.show()

#Ejemplo 5: ajustamos un polinomio para p=3,m=1000
m=1000
p=2   
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=1000,p=3')
plt.legend()
plt.show()
#Ejemplo 6: ajustamos un polinomio para p=4,m=1000
m=1000
p=3
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=1000,p=4')
plt.legend()
plt.show()
#Ejemplo 7: ajustamos un polinomio para p=6,m=1000
m=1000
p=5  
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=1000,p=6')
plt.legend()
plt.show()
#Ejemplo 8: ajustamos un polinomio para p=100,m=1000
m=1000
p=100   
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=1000,p=100')
plt.legend()
plt.show()
#Ejemplo 9: ajustamos un polinomio para p=3,m=10000
m=10000
p=2   
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=10000,p=3')
plt.legend()
plt.show()
#Ejemplo 10: ajustamos un polinomio para p=4,m=10000
m=10000
p=3   
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=10000,p=4')
plt.legend()
plt.show()
#Ejemplo 11: ajustamos un polinomio para p=6,m=10000
m=10000
p=5   
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=10000,p=6')
plt.legend()
plt.show()
#Ejemplo 12: ajustamos un polinomio para p=100,m=10000
m=10000
p=100  
pol1=np.zeros(50)

plt.plot(Regpol(m,p)[3],Regpol(m,p)[4],'r--',label='Datos reales')
plt.plot(Regpol(m,p)[3],Regpol(m,p)[0],'b.',label='Datos observados')
plt.plot(a,pol(p,Regpol(m,p)[2],pol1),'g',label='Ajuste polinomial')
plt.grid()
plt.xlabel('Valores de los regresores x')
plt.ylabel('Valores ajustados')
plt.title('Gráfica de ajuste para m=10000,p=100')
plt.legend()
plt.show()

"""Finalmente, comparamos los algoritmos QR para el caso m=1000,p= 1 hasta 200"""

from scipy import linalg as la
import time #Librería para medir tiempos

#QR para p=200,m=1000
m=3000 
p=300
#Creamos un vector con las entradas entre 1 y 200 para graficar los tiempos
k=np.arange(0,p)
#Implementamos ambos algoritmos Y medimos tiempo.
 
times_QR=np.zeros(p,dtype=float)
times_QRSci=np.zeros(p,dtype=float).copy() #creamos dos vectores vacíos para guardar los 100 tiempos de ejecución


for i in range(1,p):
    A=Regpol(m,i)[1] #Es la matriz de vandermonde X
    #Registramos el primer tiempo de QR implementado
    T0L = time.time()
    GMmod(A)
    TfL = time.time()
    #Registramos el último tiempo de QR implementado
    #Registramos el primer tiempo de QR de Scipy
    T0C = time.time()
    la.qr(A,mode='economic')
    TfC = time.time()
    #Registramos el último tiempo de Qr implementado
    #Guardamos los tiempos de ejecución
    times_QR[i]=TfL-T0L
    times_QRSci[i]=TfC-T0C  


#Graficamos los tiempos colocando una rejilla, etiquetas y títulos.
plt.plot(k,times_QR, label="Algoritmo QR implementado")
plt.plot(k,times_QRSci, label= "Algoritmo QR de scipy")
plt.grid()
plt.xlabel('Grado del polinomio')
plt.ylabel('Tiempo')
plt.title('Comparación del tiempo de ejecución de los dos algoritmos')
plt.legend()
plt.show()
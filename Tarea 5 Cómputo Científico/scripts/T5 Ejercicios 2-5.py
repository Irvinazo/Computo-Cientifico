"""Importamos las librerías necesarias"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import time

"""--------------------- Ejercicio 2: implementar la congruencia del ejercicio para generar variables Uniformes(0,1) ---------------------"""

"Esta es una función que genera n 'variables aleatorias' Uniformes(0,1) utilizando el algoritmo del ejercicio 2 de la tarea"

def gen_unif(n,arg):                                           
                                                                #Vamos a usar una función de dos argumentos, n es el número de muestras y arg será 
    semilla=time.time()                                         #un argumento que, si no es 1, generará algo aleatorio, y si es 1, generará algo determinista.
    semilla=int(semilla)                                        #Creamos un valor semilla a partir del tiempo cibernético de la computadora. Lo volvemos entero.
    Z=np.zeros(n,dtype=float)                                    
    muestra=np.zeros(n,dtype=float)                             #Creamos dos vectores vacíos auxiliares

    if arg==1:                                                  #Si el argumento es 1, entonces no habrá aleatoriedad, y haremos la congruencia
        X_0=np.array([1111,3452,5642,5346,2454],dtype=float)    #con este vector de 5 entradas escogidas al azar pero fijas. Serán las primeras entradas del 
        Z[0:5]=X_0                                              #vector Z con el que haremos las congruencias
        muestra[0:5]=X_0/(2**31-1)                              #En el vector muestra, metemos tales entradas de X_0 y las normalizamos por 2**31-1
        for i in range(5,n):
            Z[i]=(107374182*Z[i-1]+104420*Z[i-5])%(2**31-1)     #A cada paso entre 5 y n, definimos x_i como el resultado de la congruencia según esta ecuación utilizando
            muestra[i]=Z[i]/(2**31-1)                           #el elemento en el índice i-1 y el índice i-5 en Z. Esto se repite a cada paso.
    else:
        X_0=np.array([semilla,3452,5642,5346,2454],dtype=float) #Si el argumento es algo distinto de 1 (tiene que haber algún argumento), realizamos 
        Z[0:5]=X_0                                              #exactamente la misma congruencia pero con la diferencia que la primera entrada del 
        muestra[0:5]=X_0/(2**31-1)                              #vector inicial X_0 será aleatorio, y por lo tanto a cada paso tendremos un resultado distinto
        for i in range(5,n):                                    
            Z[i]=(107374182*Z[i-1]+104420*Z[i-5])%(2**31-1)
            muestra[i]=Z[i]/(2**31-1)
    return(muestra)

"""Ejemplo 1: probamos el algoritmo para n= 100 000, sin aleatoriedad (el argumento de la función es 0, y por lo tanto no hay aleatoriedad)"""

Z=gen_unif(100000,1)                                           #Generamos nuestra muestra no aleatoria

plt.hist(Z,color='red',alpha=0.5,rwidth=0.95,density=True)
plt.plot(np.linspace(0,1,100),np.ones(100),'b',lw=1)
plt.title('Histograma creado con la congruencia')
plt.xlabel('Valores de la muestra')
plt.ylabel('Frecuencia')
plt.legend(('Densidad de U(0,1)','Hist. de muestras'))
plt.show()                                                      #Graficamos el histograma de las realizaciones

np.mean(Z)                                                      #Finalmente hallamos dos estadísticos con nuestra muestra. Uno de ellos es la media     
np.var(Z)                                                       #otro la varianza, que en el caso de una uniforme en (0,1), son 1/2 y 1/12=0.083333 approx., respectivamente

"""Ejemplo 2: probamos el algoritmo para n=100 000, con aleatoriedad"""

Z=gen_unif(100000,0)                                           #Generamos nuestra muestra aleatoria en cada realización

plt.hist(Z,color='red',alpha=0.5,rwidth=0.95,density=True)
plt.plot(np.linspace(0,1,100),np.ones(100),'b',lw=1)
plt.title('Histograma creado con la congruencia')
plt.xlabel('Valores de la muestra')
plt.ylabel('Frecuencia')
plt.legend(('Densidad de U(0,1)','Hist. de muestras'))
plt.show()                                                      #Graficamos el histograma de las realizaciones

np.mean(Z)                                                      #Hallamos la media y la varianza    
np.var(Z)                                                       


"""--------------------- Ejercicio 5: implementar el algoritmo Adaptive Rejection Sampling y simular una $Gamma(2,1)$, 10,000 muestras. ---------------------"""

#Estudiamos la variable aleatoria Gamma(2,1). 

scipy.stats.gamma.ppf(.99,2)
x=np.linspace(0,7,100)
y=x*np.exp(-x)
plt.grid()
plt.plot(x,y,'r--')
plt.show()




"""Función de la log densidad de la gamma(2,1)"""
def loggamma(x):
    g=np.log(x*np.exp(-x))
    return(g)

"""Función que crea la recta entre dos puntos (x1,y1),(x2,y2) y que la evalúa en x"""
def recta(y2,y1,x2,x1,x):
    recta=((y2-y1)/(x2-x1))*(x-x1)+y1
    return(recta)

#Seleccionamos 4 puntos iniciales 
s=np.array([0.5,3,5,7])
len(s)
#El último punto inicial se tomó utilizando un cuantil 0.99, que es aproximadamente de 6.5, por lo que entre 0 y 7 se encuentra la mayor
#de la masa de la variable gamma.

#Aplicamos la función loggamma al vector s inicial.
logs=loggamma(s)

"""Función envolvente. Dicha función calcula el valor de la envolvente superior 
de la función loggamma de acuerdo a las rectas creadas por un vector muestral"""

def env(S,x):
    r=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]<S[0]:
            r[i]=recta(loggamma(S)[1],loggamma(S)[0],S[1],S[0],x[i])
        elif S[len(S)-1]<=x[i]:
            r[i]=recta(loggamma(S)[len(S)-1],loggamma(S)[len(S)-2],S[len(S)-1],S[len(S)-2],x[i])
        elif S[0]<=x[i] and x[i]<S[1]:
            r[i]=recta(loggamma(S)[2],loggamma(S)[1],S[2],S[1],x[i])
        elif S[len(S)-2]<=x[i] and x[i]<S[len(S)-1]:
            r[i]=recta(loggamma(S)[len(S)-2],loggamma(S)[len(S)-3],S[len(S)-2],S[len(S)-3],x[i])
        else: 
            m=np.where(x[i]>S)[0][0] 
            w=np.array([recta(loggamma(S)[m+1],loggamma(S)[m],S[m+1],S[m],x[i]),recta(loggamma(S)[m-1],loggamma(S)[m-2],S[m-1],S[m-2],x[i])])
            r[i]=np.min(w)
    return(r)

#Visualizamos la envolvente:
x=np.linspace(0.1,7,1000)
y=x*np.exp(-x)
z=env(s,x)
plt.grid()
plt.plot(x,loggamma(y),'r--')
plt.plot(x,z,'g')
plt.show()

g=np.exp(env(s,x))


def e_env(S,x):
    z=np.exp(env(S,x))
    return(z)

#Integramos g numéricamente para obtener la constante de normalización de $g$:

from scipy import integrate
S=np.array([0.5,3,5,7])
c=integrate.quad(e_env,0,20,args=(S))






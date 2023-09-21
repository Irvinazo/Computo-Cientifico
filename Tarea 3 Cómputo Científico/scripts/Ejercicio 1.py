"""Importamos las paqueterías necesarias"""
import numpy as np
from scipy.stats import uniform
from scipy.stats import norm 
from scipy.linalg import cholesky

"""Importamos descomposición QR"""
def GMmod(A):                                     
                                                  #Definimos la función
    v=A.copy()
    v=v.astype(np.float64)                        #Creamos una copia de la matriz A para trabajar, la volvemos flotante.
    R=np.zeros((np.shape(A)[1],np.shape(A)[1]))   
    Q=np.zeros(np.shape(A))                       #Creamos las matrices Q y R en donde almacenaremos la factorización 

                                                  #Implementamos el algoritmo:

    for i in range(len(R)): #Para cada una de las columnas de A:
        R[i,i]=np.sqrt((v[:,i]).T@v[:,i]) #extraemos la norma de esos vectores
        Q[:,i]=v[:,i]/R[i,i] # Y a las columnas de Q, se les añade el vector v_i normalizado
        for j in range(i+1,len(R)): #para la columna i fija, a partir del renglón entre i hasta el último renglón,
            R[i,j]=(Q[:,i]).T@v[:,j] #Rij lo rellenamos
            v[:,j]=v[:,j]-R[i,j]*Q[:,i] #actualizamos v

    return[A,Q,R]       #Nuevamente pedimos que el algoritmo nos regrese la terna A,Q,R, para recordar la forma de la ecuación A=QR. 

"""Importamos descomposición de Cholesky"""
def CholFact(A):
    R=A.astype(np.float64) #volvemos nuevamente A de entradas flotantes
    for k in range(len(A)): #realizamos el algoritmo del Trefethen
        for j in range(k+1,len(A)):
            R[j,j:]=R[j,j:]-R[k,j:]*R[k,j]/R[k,k] #Observación: no utilizamos el conjugado de R[k,j] aquí, pues estaremos trabajando con matrices reales
        R[k,k:]=R[k,k:]/np.sqrt(R[k,k]) 
    
    return[R]

"""Función para recuperar la matriz original a partir de Cholesky:"""
def ReChol(A):
    X=A.copy()
    for i in range(len(A)):
        for j in range(i):
           X[i][j]=0
    return(X)
 #Creamos primero la matriz unitaria Q.Fijamos una semilla fija para repetir resultados, n=10.
 
np.random.seed(10)                              #Fijamos una semilla n=10
U=uniform.rvs(loc=-5,scale=10,size=400)         #Creamos una vector aleatorio en [-5,5].
A=np.reshape(U,(20,20))                         #Lo volvemos matriz de 20x20.
Q=GMmod(A)[1]                                   #Realizamos descomposición QR de A, y seleccionamos Q, que es matriz unitaria.

#Generamos ahora una matriz de ceros pero con diagonal que contenga 20 errores normales N(0,0.02)
err=np.diagflat(norm.rvs(loc=0,scale=0.02,size=20))

 #Vamos a construir dos matrices: B1 bien condicionada y B2 mal condicionada, junto con sus respectivas matrices de errores.

"""Matriz "bien" condicionada B1"""

a1=1.1                                           #Utilizamos a1=1.1 para el caso "bien" condicionado.
eig1=np.zeros(20)                                #Creamos artificialmente 20 eigenvalores que formen una sucesión geométrica
for i in range(0,20):                            #Forzamos a que el eigenvalor más chico sea 1
    eig1[i]=(a1**(20))/(a1**(i+1))               #En este caso, K(B1) será approx. 6.1159.

D1=np.diag(eig1)                                 #Construimos una matriz diagonal con los eigenvalores anteriores.
B1=Q.T@(D1@Q)                                    #Construimos la matriz B1 "bien" condicionada y la matriz B1 con error en los eigenvalores
B1_err=Q.T@((D1+err)@Q)                          #K(B1_err) es de approx 6.1017 con estos valores de la semilla

"""Matriz mal condicionada B2"""

a2=7                                           #Utilizamos un coeficiente a2=7
eig2=np.zeros(20)                                #Creamos artificialmente otros 20 eigenvalores que formen otra sucesión geométrica
for i in range(0,20):
     eig2[i]=(a2**(20))/(a2**(i+1))               #Forzamos a que la matriz tenga eigenvalor 1. En este caso, K(B2)approx 1.13988e+16

D2=np.diag(eig2)                                 #Construimos una matriz diagonal con estos eigenvalores      
B2=Q.T@(D2@Q)                                    #Construimos la matriz B2 mal condicionada y su matriz asociada con el error
B2_err=Q.T@((D2+err)@Q)                          #En este caso, K(B2_err) es de approx. 1.1343e+16

#Con lo anterior, procedemos a utilizar la descomposición de Cholesky de
#Las matrices B1, B2, B1_err y B2_err.

"""Descomposición de B1 y B1_err"""
Z=CholFact(B1)[0]
W=CholFact(B1_err)[0]                       #Extraemos las descomposiciones de Cholesky de B1 y B1_err
Zc=ReChol(Z)
Wc=ReChol(W)                                #Aplicamos la función ReChol, para anular los términos debajo de
                                             #la diagonal, los cuales no son útiles para los fines de Cholesky

"""Descomposición de B2 y B2_err"""

X=CholFact(B2)[0]
Y=CholFact(B2_err)[0]
Xc=ReChol(X)
Yc=ReChol(Y)                                #Análogo a lo hecho antes
"""Descomposición de B2 y B2_err usando Cholesky de scipy """

R=cholesky(B2)
S=cholesky(B2_err)

"""Realizamos varias pruebas de comparación:"""
#Prueba 1: hallar el valor de la entrada más grande en valor absoluto de la diferencia
"""Caso bien condicionado, Cholesky implementado"""
np.max(abs(Zc-Wc))
"""Caso mal condicionado, Cholesky implementado"""
np.max(abs(Xc-Yc))

#Prueba 2: hallar la norma inducida 2 de la diferencia:
"""Caso bien condicionado, Cholesky implementado"""
np.linalg.norm(Zc-Wc,ord=2)
"""Caso mal condicionado, Cholesky implementado"""
np.linalg.norm(Xc-Yc,ord=2)

#Prueba 3: hallar con la norma del supremo qué tanta diferencia hay entre la recuperación y la original
"""Caso bien condicionado, Cholesky implementado"""
np.max(abs(Zc.T@Zc-B1))
np.max(abs(Wc.T@Wc-B1))
"""Caso mal condicionado, Cholesky implementado"""
np.max(abs(Xc.T@Xc-B2))
np.max(abs(Yc.T@Yc-B2))

#Prueba 4: hallar con la norma inducida 2 la diferencia entre las matrices anteriores
"""Caso bien condicionado, Cholesky implementado"""
np.linalg.norm(Zc.T@Zc-B1,ord=2)
np.linalg.norm(Wc.T@Wc-B1,ord=2)
"""Caso mal condicionado, Cholesky implementado"""
np.linalg.norm(Xc.T@Xc-B2,ord=2)
np.linalg.norm(Yc.T@Yc-B2,ord=2)

#####
#Prueba 1: hallar el valor de la entrada más grande en valor absoluto de la diferencia
"""Caso mal condicionado, Cholesky de scipy"""
np.max(abs(R-S))
"""Caso mal condicionado, Cholesky implementado"""
np.max(abs(Xc-Yc))

#Prueba 2: hallar la norma inducida 2 de la diferencia:
"""Caso mal condicionado, Cholesky de scipy"""
np.linalg.norm(R-S,ord=2)
"""Caso mal condicionado, Cholesky implementado"""
np.linalg.norm(Xc-Yc,ord=2)

#Prueba 3: hallar con la norma del supremo qué tanta diferencia hay entre la recuperación y la original
"""Caso mal condicionado, Cholesky de scipy"""
np.max(abs(R.T@R-B2))
np.max(abs(S.T@S-B2))
"""Caso mal condicionado, Cholesky implementado"""
np.max(abs(Xc.T@Xc-B2))
np.max(abs(Yc.T@Yc-B2))

#Prueba 4: hallar con la norma inducida 2 la diferencia entre las matrices anteriores
"""Caso mal condicionado, Cholesky de scipy"""
np.linalg.norm(R.T@R-B2,ord=2)
np.linalg.norm(S.T@S-B2,ord=2)
"""Caso mal condicionado, Cholesky implementado"""
np.linalg.norm(Xc.T@Xc-B2,ord=2)
np.linalg.norm(Yc.T@Yc-B2,ord=2)

"""Medición de tiempos, scipy vs implementación"""

import time #Librería para medir tiempos

p=500
#Registramos el primer tiempo de scipy
T0s = time.time()
for i in range(p):
    cholesky(B2)    
Tfs = time.time() #Registramos el último tiempo de scipy
    
#Registramos el primer tiempo del implementado
T0C = time.time()
for i in range(p):
    CholFact(B2)  
TfC = time.time()#Registramos el último tiempo del implementado    

#comparamos los tiempos
Tfs-T0s
TfC-T0C



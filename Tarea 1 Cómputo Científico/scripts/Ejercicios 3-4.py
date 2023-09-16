import numpy as np
from scipy.stats import uniform

#Este es el algoritmo LUP
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
#Este es el algoritmo backward substitution
def backsub(M,a):
    
    A=M.astype(np.float64)
    b=a.astype(np.float64)
    x=np.empty(len(A),dtype=float)

    for k in range(len(A)):
        v=np.empty(k,dtype=float)
        v[0:k]=A[len(A)-1-k][len(A)-1-k+1:len(A)]*x[len(A)-1-k+1:len(A)]
        x[len(A)-1-k]=(b[len(A)-1-k]-sum(v))/A[len(A)-1-k][len(A)-1-k]

    return(x)
#Este es el algoritmo forward substitution
def forsub(M,a):
    
    A=M.astype(np.float64)
    b=a.astype(np.float64)
    x=np.empty(len(A),dtype=float)

    for k in range(len(A)):
        v=np.empty((k),dtype=float)
        v[0:k]=A[k][0:k]*x[0:k]
        x[k]=(b[k]-sum(v))/A[k][k]
    return(x)

"""Ejercicio 3"""
"""Ejemplo 1: dar la descomposición de una matriz aleatoria con entradas aleatorias U(0,1) de tamaño 5x5:"""


np.random.seed(10) #Colocamos una semilla para replicar resultados
M=uniform.rvs(size=25) 
M=np.reshape(M,(5,5)) #creamos un vector de 25 entradas uniformes en (0,1) aleatorio, y lo volvemos matriz

LUPfact(M)
PM,M,LM,UM=LUPfact(M)[0:4] #realizamos la descomposición LUP de M
PM
M
LM
UM #A las 4 salidas de la función las renombramos y las colocamos aquí para poder imprimirlas en la terminal de la forma más cercana a PA=LU

"""Ejemplo 2: dar la descomposición LUP de la siguiente matriz:"""

A=np.array([[1,0,0,0,1],[-1,1,0,0,1],[-1,-1,1,0,1],[-1,-1,-1,1,1],[-1,-1,-1,-1,1]],dtype=float) #Cargamos A

PA,A,LA,UA=LUPfact(A)[0:4] #descomponemos con LUP
PA
A
LA
UA #Realizamos el mismo procedimiento que antes

"""Ejercicio 4"""
"""Ejemplo 1: usando la descomposición LUP del sistema aleatorio, resolver Dx=b para b vector uniforme en (0,1), con la matriz D=M"""

#Colocamos una semilla y generamos los vectores b (los pensaremos como las columnas de la matriz B, la cual creamos como hicimos con M)
np.random.seed(5) 
B=uniform.rvs(size=25)
B=np.reshape(B,(5,5))

#Una vez descompuesta M y desempaquetada, procedemos a resolver los sistemas Ly=Pb y Ux=y para resolver Mx=b
#Lo haremos para cada uno de los 5 casos, y guardamos el resultado en la matriz XM. Pensamos en las respuestas x como las columnas de X, tal y como se 
#haría en matemáticas de manera usual.
XM=np.zeros((5,5),dtype=float)
for k in range(5):

    #Resolvemos Ly=Pb:
    y=forsub(LM,PM@B[:,k])
    #Resolvemos Ux=y
    x=backsub(UM,y)

    #Guardamos en una matriz de respuestas (las respuestas son vectores columna)
    XM[:,k]=x

#Podemos verificar que en efecto el producto de las matrices nos da como respuesta la matriz producto
M@XM
B

"""Ejemplo 2: usando la descomposición LUP, resolvemos Dx=b, con b aleatorio, para b vector uniforme, con la matriz D=A"""

#Colocamos una semilla y generamos los vectores aleatorios B uniformes en (0,1). Los pensamos como columnas.
np.random.seed(15)
B=uniform.rvs(size=25)
B=np.reshape(B,(5,5))
B

#Una vez descompuesta A y desempaquetada, procedemos a resolver los sistemas Ly=Pb y Ux=y para resolver Ax=b.

#Lo haremos para cada uno de los casos
XA=np.zeros((5,5),dtype=float)
XA
for k in range(5):

    #Resolvemos Ly=Pb:
    y=forsub(LA,PA@B[:,k])
    #Resolvemos Ux=y
    x=backsub(UA,y)

    #Guardamos en una matriz de respuestas y verificamos
    XA[:,k]=x
    
#Verificamos que coincidan
A@XA
B

import numpy as np

"""La primera parte de este script se compone de los algoritmos Backward y Forward para la solución
de sistemas de ecuaciones triangulares superiores e inferiores respectivamente.
"""
#Algoritmo Backward substitution:
def backsub(M,a):
    
    A=M.astype(np.float64) #Volvemos la matriz A y el vector b como flotantes para evitar problemas de enteros
    b=a.astype(np.float64)
    x=np.empty(len(A),dtype=float) #creamos un vector vacío  de tamaño igual a la dimensión de la matriz A

    for k in range(len(A)):
        v=np.empty(k,dtype=float) #creamos un vector vacío para guardar las sumas parciales
        v[0:k]=A[len(A)-1-k][len(A)-1-k+1:len(A)]*x[len(A)-1-k+1:len(A)]
        x[len(A)-1-k]=(b[len(A)-1-k]-sum(v))/A[len(A)-1-k][len(A)-1-k] #Realizamos el algoritmo backward del Trefethen

    return(x) #Devolvemos el vector de soluciones x

#Algoritmo Forward substitution:
def forsub(M,a):
    
    A=M.astype(np.float64) #Volvemos la matriz A y el vector b como flotantes para evitar problemas de enteros
    b=a.astype(np.float64)
    x=np.empty(len(A),dtype=float) #creamos un vector vacío  de tamaño igual a la dimensión de la matriz A


    for k in range(len(A)):
        v=np.empty((k),dtype=float) #creamos un vector vacío para guardar las sumas parciales
        v[0:k]=A[k][0:k]*x[0:k]
        x[k]=(b[k]-sum(v))/A[k][k] #Realizamos el algoritmo forward del Trefethen
    return(x)

"""La segunda parte de este script contiene el algoritmo para la factorización LUP"""

def LUPfact(A):


    U=A.astype(np.float64) #Nuevamente volvemos la matriz como con entradas flotantes 
    L=np.identity(len(A))
    P=np.identity(len(A)) #Creamos dos matrices L y P de identidades, distintas, para ir guardando las operaciones de permutación en P y la t. inferior en L

    for k in range(len(A)):
        m=max(np.abs(U[k:,k]))
        pm=np.where(np.abs(U[k:,k])==m)
        pm=pm[0][0]+k #Hallamos el índice en donde se alcanza el máximo en valor absoluto debajo de la diagonal de la matriz A

        L[pm][0:k], L[k][0:k]=L[k][0:k].copy(), L[pm][0:k].copy()
        U[pm][k:], U[k][k:]=U[k][k:].copy(), U[pm][k:].copy()
        P[pm], P[k]=P[k].copy(), P[pm].copy() #Actualizamos los renglones de las matrices A, L y P permutándolas respectivamente
        for i in range(k+1,len(A)):
            L[i][k]=U[i][k]/U[k][k]         
            U[i][k:len(A)]=U[i][k:len(A)]-L[i][k]*U[k][k:len(A)] #realizamos el algoritmo LUP del Trefethen

    return[P,A,L,U] #Regresamos con esta particular estructura la respuesta del algoritmo, para obtener lo más parecido a la forma PA=LU de la descomposición

"""La tercera parte de este script contiene el algoritmo para la factorización de Cholesky de 
una matriz hermitiana definida positiva"""

def CholFact(A):
    R=A.astype(np.float64) #volvemos nuevamente A de entradas flotantes
    for k in range(len(A)): #realizamos el algoritmo del Trefethen
        for j in range(k+1,len(A)):
            R[j,j:]=R[j,j:]-R[k,j:]*R[k,j]/R[k,k] #Observación: no utilizamos el conjugado de R[k,j] aquí, pues estaremos trabajando con matrices reales
        R[k,k:]=R[k,k:]/np.sqrt(R[k,k]) 
    
    return[R]



"""
Este es el algoritmo para 'solucionar' un sistema de la forma $Ax=b$, donde A es una matriz de tamaño mxn, m>=n,
donde la palabra solucionar se usa en el sentido de encontrar la solución x que minimice la distancia $Ax-b$ en norma 2.
En particular y de acuerdo a lo visto en clase, este algoritmo puede encontrar el estimador de mínimos cuadrados
en un problema de regresión.
"""

import numpy as np #Importamos la paquetería numpy.

#Importamos la función para la factorización QR de una matriz A de mxn.

def GMmod(A):
    v=A.copy()
    v=v.astype(np.float64)
    R=np.zeros((np.shape(A)[1],np.shape(A)[1]))
    Q=np.zeros(np.shape(A))

    for i in range(len(R)): 
        R[i,i]=np.sqrt((v[:,i]).T@v[:,i]) 
        Q[:,i]=v[:,i]/R[i,i] 
        for j in range(i+1,len(R)): 
            R[i,j]=(Q[:,i]).T@v[:,j] 
            v[:,j]=v[:,j]-R[i,j]*Q[:,i] 

    return[A,Q,R]

#Importamos Back substitution.

def backsub(M,a):
    
    A=M.astype(np.float64)
    b=a.astype(np.float64)
    x=np.empty(len(A),dtype=float)

    for k in range(len(A)):
        v=np.empty(k,dtype=float)
        v[0:k]=A[len(A)-1-k][len(A)-1-k+1:len(A)]*x[len(A)-1-k+1:len(A)]
        x[len(A)-1-k]=(b[len(A)-1-k]-sum(v))/A[len(A)-1-k][len(A)-1-k]

    return(x)

"""Algoritmo para 'resolver' un sistema de ecuaciones Ax=y, vía factorización QR."""

def QRsol(A,b):      #Definimos el nombre del algoritmo.
    Q=GMmod(A)[1]    
    R=GMmod(A)[2]    #De la terna A,Q,R, que nos devuelve el algoritmo QR, extraemos Q y R.
    Qb=Q.T@b         #Realizamos la multiplicación Q*b=Py.
    x=backsub(R,Qb)  #Resolvemos el sistema Rx=Q*b con el algoritmo back substitution.
    return(x)        #Regresamos la solución.




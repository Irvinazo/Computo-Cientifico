import numpy as np
from scipy.linalg import eigvals
"""Cargamos el algoritmo de la factorización QR"""

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

"""Implementamos el algoritmo QR con shift para calcular los eigenvalores con n iteraciones en el algoritmo"""


def QRshift(B,n):
    A=B.copy()                          #Creamos una copia de la matriz B
    Q=np.zeros((len(B),len(B)))         
    R=np.zeros((len(B),len(B)))         #Creamos 3 matrices, Q, R y Id para las iteraciones.
    Id=np.identity(len(B))
    for i in range(n):                  
        s=A[len(B)-1,len(B)-1]          #A cada paso, este algoritmo extrae la entrada inferior derecha más extrema de B.
        Q=GMmod(A-s*Id)[1]              
        R=GMmod(A-s*Id)[2]              #Realizamos la descomp. QR a la matriz A con el shift correspondiente s*Id.
        A=R@Q+s*Id                      #Guardamos las matrices Q y R de la descomp. Realizamos el producto R@Q, regresamos el shift y actualizamos A

    eig=np.zeros(len(A))                #Adicionalmente (y dado que es el propósito de este algoritmo), creamos un vector
    for i in range(len(A)):             #que guarde los eigenvalores aproximados, los cuales están en la diagonal de A luego de las n iteraciones.
        eig[i]=A[i,i]
    return(A,Q,eig)                     #Regresamos la matriz A casi diagonal, Q que aproxima a la base de eigenvectores, y el vector de eigenvalores.

#Ejemplo: A es una matriz que fue construida ad hoc. Sus eigenvalores son 2 y 1 respectivamente.

A=np.array([[11,-3],[2,4]],dtype=float)/5           #Construimos A
QRshift(A,8)[0]                                 
QRshift(A,8)[1]                                     #Realizamos el algoritmo con shift para A con 8 iteraciones. 
QRshift(A,8)[2]                                     #Obtenemos A, Q y el vector de eigenvalores


"""--------------------------------------- Ejercicio 2: implementamos el algoritmo QR con shift para la matriz del ejercicio 1---------------------------"""

#Construimos las 4 matrices con epsilon=10^{-N}, N=1,3,4,5:

B1=np.array([[8,1,0],[1,4,10**(-1)],[0,10**(-1),1]],dtype=float)
B3=np.array([[8,1,0],[1,4,10**(-3)],[0,10**(-3),1]],dtype=float)
B4=np.array([[8,1,0],[1,4,10**(-4)],[0,10**(-4),1]],dtype=float)
B5=np.array([[8,1,0],[1,4,10**(-5)],[0,10**(-5),1]],dtype=float)

#Aplicamos QR y obtenemos los eigenvalores de las matrices.
A1=QRshift(B1,50)[0]
A3=QRshift(B3,75)[0]
A4=QRshift(B4,75)[0]
A5=QRshift(B5,75)[0]

A1
A3
A4
A5

#Corroboramos que los eigenvalores hallados antes son los de la matriz X siguiente usando scipy

X=np.array([[8,1,0],[1,4,0],[0,0,1]],dtype=float)

eigvals(X)

"""-----------------------------Ejercicio 5: corroboración de que una matriz ortogonal queda inalterada bajo QR sin shift---------------------"""


"""Implementamos el algoritmo QR sin shift para calcular los eigenvalores haciendo n iteraciones en el algoritmo"""

def QRdiag(B,n):
    A=B.copy()
    for i in range(n):
        A=GMmod(A)[2]@GMmod(A)[1]
    return(A)

#Creamos una matriz ortogonal ad hoc:
M=(np.array([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[-1/np.sqrt(6),2/np.sqrt(6),-1/np.sqrt(6)],[1/np.sqrt(2),0,-1/np.sqrt(2)]])).T
M
QRdiag(M,2)


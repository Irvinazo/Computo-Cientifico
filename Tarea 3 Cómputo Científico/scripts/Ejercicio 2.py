import numpy as np
from scipy.stats import norm, uniform
from scipy.linalg import inv 

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

"""Importamos Backward Substitution"""
def backsub(M,a):
    
    A=M.astype(np.float64)
    b=a.astype(np.float64)
    x=np.empty(len(A),dtype=float)

    for k in range(len(A)):
        v=np.empty(k,dtype=float)
        v[0:k]=A[len(A)-1-k][len(A)-1-k+1:len(A)]*x[len(A)-1-k+1:len(A)]
        x[len(A)-1-k]=(b[len(A)-1-k]-sum(v))/A[len(A)-1-k][len(A)-1-k]

    return(x)

"""Importamos la solución de sistemas vía QR"""
def QRsol(A,b):
    Q=GMmod(A)[1]
    R=GMmod(A)[2]
    Qb=Q.T@b
    x=backsub(R,Qb)
    return(x)


"""
Buscamos resolver el problema de mínimos cuadrados y=XB+E, e_i~N(0,\sigma)
usando la implementación de la descomposición QR; B es dx1, X es nxd

d=5, n=20, beta=(5,4,3,2,1)^t, sigma=0.13
"""

"""Ejercicio 2a)"""

#Creamos una matriz X con entradas aleatorias U(0,1) de tamaño nxd=20x5:
#Fijamos nuestra semilla en n=15
np.random.seed(15)
X=uniform.rvs(size=100)
X=np.reshape(X,(20,5))

#Creamos ahora la matriz Delta con entradas ~ Normal(0,0.01)
Delta=norm.rvs(scale=0.01,size=100)
Delta=np.reshape(Delta,(20,5))

#Guardamos la matriz X+DeltaX en la matriz Xerr (la matriz con error)
Xerr=X+Delta

#Por curiosidad, extraemos el número de condición de las nuevas matrices.
#Su tamaño nos indica si estas matrices están bien o mal condicionadas
np.linalg.cond(X)
np.linalg.cond(Xerr)

#Creamos un vector de errores e, con distribución normal
#de media 0 y desviación estándar de 0.13:
e=norm.rvs(scale=0.13,size=20)

#Cargamos el vector beta
beta=np.array([5,4,3,2,1],dtype=float)

#Simulamos el vector de observaciones y a partir de la matriz X original
y=X@beta+e

#Encontramos el estimador de mínimos cuadrados \hat{\beta} usando
#la descomposición QR de X, y resolviendo el sistema
#QRb=y sii Rb=Q*y. Resolvemos y hallamos b:

#Estimador de mínimos cuadrados para X
bhat=QRsol(X,y)

#Encontramos el estimador de mínimos cuadrados para 
#la matriz X+Delta (la matriz con ruido)
bhat_p=QRsol(Xerr,y)

#Construimos ahora el estimador teórico bhat_c=(X^tX)^{-1} X^t y, usando matriz inversa de scipy
bhat_c=(inv(Xerr.T@Xerr)@Xerr.T)@y

#Comparamos
bhat 
bhat_p
bhat_c


"""Ejercicio 2b)"""

#Realizamos ahora el mismo experimento pero con una matriz X mal condicionada,
#para ello, provocaremos casi colinealidad en todas las columnas

Y=X.copy()                       #Creamos una copia de X con entradas uniformes
for i in range(5):               #tomamos la columna 1 de X, y la copiamos en el resto de columnas
    Y[:,i]=((i+1)/3)*X[:,0]      #salvo factores escalares en las entradas de Y

shake=uniform.rvs(loc=0,scale=0.0001,size=100) #Para evitar colinealidad absoluta, alteramos 
shake=np.reshape(shake,(20,5))                    #ligeramente la matriz Y con un ruido uniforme
                                    
Y=Y+shake           #Renombramos nuestra matriz Y, ya con el ruido. Esta matriz casi presenta colinealidad.
Yerr=Y+Delta        #Aprovechamos la distorsión normal antes creada, Delta, para crear Y+DeltaY=Yerr

#Extraemos el número de condición de las nuevas matrices
np.linalg.cond(Y)
np.linalg.cond(Yerr)

#Simulamos ahora las nuevas observaciones con la matriz Y y las guardamos con el nombre de z
z=Y@beta+e

#Procedemos a encontrar los estimadores de mínimos cuadrados.
#Estimador de mínimos cuadrados para Y
bmhat=QRsol(Y,z)

#Encontramos el estimador de mínimos cuadrados para 
#la matriz Yerr (la matriz con ruido)
bmhat_p=QRsol(Yerr,z)

#Construimos ahora el estimador teórico bmhat_c usando inversión de matrices de scipy
bmhat_c=(inv(Yerr.T@Yerr)@Yerr.T)@z

#Comparamos
bmhat
bmhat_p
bmhat_c


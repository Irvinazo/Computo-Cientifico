import numpy as np

"""Función para extraer una base ortonormal mediante el algoritmo GM modificado de una matriz de  tamaño mxn, donde m>=n"""
def GMmod(A):                                     #Definimos la función
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


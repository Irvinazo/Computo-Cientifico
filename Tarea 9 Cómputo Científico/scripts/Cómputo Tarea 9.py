import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------- Ejercicio 1 ------------------------------------------------------------------------------"""
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

np.random.seed(10)                 #Colocamos la semilla seed=10 para este problema.

"""Creamos un vector para guardar los datos"""
data=np.array([7,7,8,8,9,4,7,5,5,6,9,8,11,7,5,5,7,3,10,3], dtype=int)

"""Creamos una función para una variable aleatoria Rademacher"""
def radem():
    u=ss.uniform.rvs(0,1)
    if u<0.5:
        r=-1
        return(r)
    else:
        r=1
        return(r)

"""Creamos una función que calcule el cociente proporcional de la objetivo f(N*,p*|x)/f(N,p|x)"""

def coc_dist_obj(Nx,px,Ny,py,data):
    Nxint=int(Nx)
    Nyint=int(Ny)
    vecx=np.zeros(Nxint,dtype=float)
    vecy=np.zeros(Nyint,dtype=float)
    for i in range(1,Nxint):
        vecx[i]=np.log(i)
    logNx=sum(vecx)
    for i in range(1,Nyint):
        vecy[i]=np.log(i)
    logNy=sum(vecy)

    s1=20*logNy-20*logNx
    s2=sum(data)*np.log(py/px)+(20*Ny-sum(data))*np.log(1-py)-(20*Nx-sum(data))*np.log(1-px)
    v=np.zeros(20,dtype=float)
    for i in range(20):    
        vecx=np.zeros(Nxint-data[i],dtype=float)
        vecy=np.zeros(Nyint-data[i],dtype=float)
        for j in range(1,Nxint-data[i]):
            vecx[j]=np.log(j)
        logNxdata=sum(vecx)
        for j in range(1,Nyint-data[i]):
            vecy[j]=np.log(j)
        logNydata=sum(vecy)
        v[i]=logNxdata-logNydata
    s3=sum(v)
    s=s1+s2+s3
    quot=np.exp(s)
    return(quot)


"""Creamos la función del algoritmo MH-Kernel Híbrido"""

def MH_gibbs(k,m,data):
    X=np.zeros((k,2),dtype=float)
    X[0]=m
    for i in range(k-1):
        y=np.zeros(2,dtype=float)
        y[0]=ss.randint.rvs(low=maxdata,high=Nmax+1)
        y[1]=ss.uniform.rvs(0,1)
        quot=coc_dist_obj(X[i,0],X[i,1],y[0],y[1],data)
        U=ss.uniform.rvs(0,1)
        if U<quot:
            X[i+1]=y                                    
        else:
            X[i+1]=X[i]
    return(X)

"""Creamos vectores iniciales m=(p,N), p~U(0,1), N~U_d(max(data),max(data)+1,...,N_{max})"""

Nmax=1000
maxdata=np.max(data)
p=ss.uniform.rvs(0,1)
N=ss.randint.rvs(low=maxdata,high=Nmax+1)
m=np.array([N,p],dtype=float)
m
k=100_000

"""Ejecutamos MH-Kerneles Híbridos"""
R=MH_gibbs(k,m,data)

plt.plot(R[:,0],R[:,1],'b',color='green',linewidth=0.3,alpha=0.9,marker='.', markersize=0.2)  #Graficamos el recorrido de la cadena 
plt.title(r'Algoritmo MH con Kerneles Híbridos.    Vector inicial $(N,p)=(538,0.77132)$ con seed=10,   100,000 iteraciones')
plt.xlabel(r'Muestreo en $N$')
plt.ylabel(r'Muestreo en $p$')
plt.grid()
plt.show()

plt.hist(R[1000:,0],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        #Graficamos los histogramas de la variable N
plt.title(r'Histograma de la variable $N$.')
plt.xlabel(r'Valor en $N$')
plt.ylabel(r'Frecuencia (normalizada)')
plt.show()

plt.hist(R[1000:,1],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        #Graficamos los histogramas de la variable p
plt.title(r'Histograma de la variable $p$.')
plt.xlabel(r'Valor en $p$')
plt.ylabel(r'Frecuencia (normalizada)')
plt.show()

np.mean(R[:,0])
np.mean(R[:,1])
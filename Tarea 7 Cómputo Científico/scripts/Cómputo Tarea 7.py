import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.special import gamma


""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""


np.random.seed(10)                 #Colocamos la semilla seed=10
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------- Ejercicio 1 ------------------------------------------------------------------------------"""
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
#Simulamos primero datos x_i con alpha=3, beta=100, considerando los casos n=4 y n=30

data4=ss.gamma.rvs(size=4,a=3,scale=1/100)
data30=ss.gamma.rvs(size=30,a=3,scale=1/100)            #Extraemos aquí las simulaciones de los datos

"""Creamos una función que nos calcula la densidad a posteriori con base la suma r2 y el producto r1 de los datos"""
def post_ej1(n,r1,r2,alpha,beta):
    if np.logical_and(1<=alpha,alpha<=4):
        if beta>0:                                           #Colocamos las indicadores en forma de condicionales
            coef=(beta**(n*alpha))/(gamma(alpha)**n)
            expon=np.exp(-beta*(r2+1))
            prod=r1**(alpha-1)
            f=coef*expon*prod                           
            return(f)
        else:
            return(0)
    else:
        return(0)                                             #Esta función nos regresa 0 justamente si caemos fuera de su soporte

"""Creamos una función que calcula la posteriori, pero suponiendo que los puntos de evaluación ya están dentro del soporte (sin indicadoras)"""
def post_ej1graf(n,r1,r2,alpha,beta):
            coef=(beta**(n*alpha))/(gamma(alpha)**n)                #El motivo de crear esta función es para construir los contour más sencillos
            expon=np.exp(-beta*(r2+1))
            prod=r1**(alpha-1)
            f=coef*expon*prod
            return(f)

""" Creamos la función que calcula el cociente de Metrópolis-Hastings con punto actual 'x' y punto futuro 'y' de la propuesta y densidad objetivo del problema"""
def mh_quotient_ej1(n,r1,r2,alphay,betay,alphax,betax):
    q1=(post_ej1(n,r1,r2,alphay,betay))/(post_ej1(n,r1,r2,alphax,betax))      #El cociente f(y)/f(x)
    #El cociente q(x|y)/q(y|x) es 1, pues la propuesta es simétrica. Aplica tanto para la propuesta 1 como para la propuesta 2
    return(q1)

""" Creamos la función del algoritmo M. Hastings, k iteraciones, n= número de datos, propuesta 1: avanzar según normales bivariadas, propuesta 2: avanzar según uniformes """
def m_hast_ej1(k,n,r1,r2,sigma1,sigma2,a='propuesta1'):
        
        if a=='propuesta1':                                 #La propuesta 1 considera una simulación de una normal bivariada

            X=np.zeros((k,2),dtype=float)                   #Matriz donde guardar las muestras
            m=np.array([3,100])                             #Mi punto inicial es (alpha, beta)=(3,100)
            X[0]=m                                          #Mi primer punto de la cadena será el punto inicial
            cov_mat=np.array([[sigma1,0],[0,sigma2]],dtype=float) #Colocamos la matriz de covarianzas para los pasos normales
            for i in range(k-1):
                y=ss.multivariate_normal.rvs(mean=X[i],cov=cov_mat,size=1)           #Simulamos una propuesta 'y' de la normal multivariada.
                quot=mh_quotient_ej1(n,r1,r2,y[0],y[1],X[i,0],X[i,1])                #Calculamos el cociente de Metrópolis-Hastings
                U=ss.uniform.rvs(0,1,size=1)                #Generamos una uniforme en (0,1)

                if U<=quot:
                    X[i+1]=y                                #Si el cociente es más grande que la uniforme, aceptamos y nos movemos a 'y'
                else:
                    X[i+1]=X[i]                             #Si el cociente es más chico que la uniforme, rechazamos y nos quedamos en 'x'
            return(X)
        
        elif a=='propuesta2':                               #La propuesta 2 considera avanzar según uniformes

            X=np.zeros((k,2),dtype=float)                   #Matriz donde guardar las muestras
            m=np.array([3,100])                             #Mi punto inicial es (alpha, beta)=(3,100)
            X[0]=m                                          #Mi primer punto de la cadena será el punto inicial
            for i in range(k-1):
                y1=ss.uniform.rvs(loc=-.03,scale=.06,size=1)          #Simulamos una propuesta para la primera entrada. La propuesta es uniforme en  [-0.03,0.03]
                y2=ss.uniform.rvs(loc=-0.5,scale=1,size=1)            #Simulamos una propuesta para la segunda entrada. La propuesta es uniforme en  [-0.5,0.5]
                quot=mh_quotient_ej1(n,r1,r2,y1+X[i,0],y2+X[i,1],X[i,0],X[i,1])                #Calculamos el cociente de Metrópolis-Hastings
                U=ss.uniform.rvs(0,1,size=1)                #Generamos una uniforme en (0,1)

                if U<=quot:
                    X[i+1,0]=y1+X[i,0]
                    X[i+1,1]=y2+X[i,1]                      #Si el cociente es más grande que la uniforme, aceptamos y nos movemos a 'y'
                else:
                    X[i+1]=X[i]                             #Si el cociente es más chico que la uniforme, rechazamos y nos quedamos en 'x'
            return(X)

 

"""---------------------------------------------- Ejemplo 1: caso n=4, propuesta 1---------------------------------------------------------------"""


r1_4=np.prod(data4)
r2_4=np.sum(data4)        #Creamos los vectores r1 y r2 asociados a estos ejemplos, donde r1 es el producto de los datos y r2 es la suma de los mismos

R1_1=m_hast_ej1(250_000,4,r1_4,r2_4,0.0005,0.5,'propuesta1') #Ejecutamos el algoritmo con 250 000 iteraciones, propuesta normal multivariada con varianzas sigma1=0.0005,sigma2=0.5

logf_1=np.log(post_ej1graf(4,r1_4,r2_4,R1_1[:10_000,0],R1_1[:10_000,1]))     #Calculamos la log-densidad y graficamos en contra del número de iteraciones
plt.plot(logf_1,color='red',alpha=0.9,linewidth=0.5)
plt.title('Gráfica Log-densidad vs número de iteraciones, n=4. Propuesta 1')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                                                               #Determinamos que lo mejor es utilizar un burn-in=3000

x_ej1=np.linspace(1, 3.5, 100)
y_ej1=np.linspace(0, 28, 100)
X_ej1,Y_ej1=np.meshgrid(x_ej1, y_ej1)
Z1=post_ej1graf(4,r1_4,r2_4,X_ej1,Y_ej1)                                  #Realizamos una rejilla de valores para evaluar la densidad y graficar curvas de nivel


plt.plot(R1_1[:,0],R1_1[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.3)   #Graficamos el recorrido de la cadena y las curvas de nivel de la densidad posterior
plt.title('Algoritmo Metrópolis-Hastings. Recorrido de la cadena. Propuesta 1: avanzar según normales. 250 000 muestras para n=4, seed=10')
plt.xlabel('Muestra en alpha')
plt.ylabel('Muestra en beta')
plt.contour(X_ej1,Y_ej1,Z1,levels=250,cmap='plasma',alpha=0.7,linewidths=0.5)
plt.show()

plt.hist(R1_1[3000:,0],density=True,bins=50,rwidth=0.95,alpha=0.8,color='b')        #Graficamos los histogramas de la variable alpha
plt.title('Histograma de la variable alpha. Caso n=4. Propuesta 1')
plt.xlabel('Valor en alpha')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R1_1[3000:,1],density=True,bins=50,rwidth=0.95,alpha=0.8,color='r')        #Graficamos los histogramas de la variable beta
plt.title('Histograma de la variable beta. Caso n=4. Propuesta 1')
plt.xlabel('Valor en beta')
plt.ylabel('Frecuencia (normalizada)')
plt.show()



"""---------------------------------------------- Ejemplo 1: caso n=4, propuesta 2---------------------------------------------------------------"""

R1_2=m_hast_ej1(250_000,4,r1_4,r2_4,0.0005,0.5,'propuesta2') #Ejecutamos el algoritmo con 250 000 iteraciones. Avanzamos con propuesta uniforme en [-0.03,0.03]x[-0.5,0.5]

logf_1_2=np.log(post_ej1graf(4,r1_4,r2_4,R1_2[:20_000,0],R1_2[:20_000,1]))     #Calculamos la log-densidad y graficamos en contra del número de iteraciones
plt.plot(logf_1_2,color='red',alpha=0.9,linewidth=0.5)
plt.title('Gráfica Log-densidad vs número de iteraciones, n=4. Propuesta 2')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                                                               #Determinamos que lo mejor es utilizar un burn-in=7000

plt.plot(R1_2[:,0],R1_2[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.3)   #Graficamos el recorrido de la cadena y las curvas de nivel de la densidad posterior
plt.title('Algoritmo Metrópolis-Hastings. Recorrido de la cadena. Propuesta 2: avanzar según uniformes. 250 000 muestras para n=4, seed=10')
plt.xlabel('Muestra en alpha')
plt.ylabel('Muestra en beta')
plt.contour(X_ej1,Y_ej1,Z1,levels=250,cmap='plasma',alpha=0.7,linewidths=0.5)
plt.show()

plt.hist(R1_2[7000:,0],density=True,bins=50,rwidth=0.95,alpha=0.8,color='b')        #Graficamos los histogramas de la variable alpha
plt.title('Histograma de la variable alpha. Caso n=4. Propuesta 2')
plt.xlabel('Valor en alpha')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R1_2[7000:,1],density=True,bins=50,rwidth=0.95,alpha=0.8,color='r')        #Graficamos los histogramas de la variable beta
plt.title('Histograma de la variable beta. Caso n=4. Propuesta 2')
plt.xlabel('Valor en beta')
plt.ylabel('Frecuencia (normalizada)')
plt.show()


"""---------------------------------------------- Ejemplo 2: n=30, propuesta 1---------------------------------------------------------------"""


r1_30=np.prod(data30)
r2_30=np.sum(data30)        #Creamos los vectores r1 y r2 asociados a estos ejemplos, donde r1 es el producto de los datos y r2 es la suma de los mismos

R2=m_hast_ej1(250_000,30,r1_30,r2_30,0.0005,0.5,'propuesta1') #Ejecutamos el algoritmo, 250 000 iteraciones. Varianzas sigma1=0.0005,sigma2=0.5


logf_2=np.log(post_ej1graf(30,r1_30,r2_30,R2[:10_000,0],R2[:10_000,1]))     
plt.plot(logf_2,color='r',alpha=0.9,linewidth=0.5)
plt.title('Gráfica Log-densidad vs número de iteraciones, n=30. Propuesta 1')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                                                         #Determinamos que lo mejor es utilizar un burn-in=3000

Z2=post_ej1graf(30,r1_30,r2_30,X_ej1,Y_ej1)                                        #Realizamos una rejilla de valores para evaluar la densidad

plt.plot(R2[:,0],R2[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.3)      #Graficamos el recorrido de la cadena, y las curvas de nivel
plt.title('Algoritmo Metrópolis-Hastings. Recorrido de la cadena. Propuesta 1: avanzar con normales. 250 000 muestras para n=30, seed=10')
plt.xlabel('Muestra en alpha')
plt.ylabel('Muestra en beta')
plt.contour(X_ej1,Y_ej1,Z2,levels=250,cmap='plasma',alpha=0.6,linewidths=0.5)
plt.grid()
plt.show()

plt.hist(R2[3000:,0],density=True,bins=50,rwidth=0.95,alpha=0.8,color='b')        #Graficamos los histogramas de la variable alpha
plt.title('Histograma de la variable alpha. Caso n=30')
plt.xlabel('Valor en alpha')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R2[3000:,1],density=True,bins=50,rwidth=0.95,alpha=0.8,color='r')        #Graficamos los histogramas de la variable beta
plt.title('Histograma de la variable beta. Caso n=30')
plt.xlabel('Valor en beta')
plt.ylabel('Frecuencia (normalizada)')
plt.show()


"""---------------------------------------------- Ejemplo 2: n=30, propuesta 2---------------------------------------------------------------"""
np.random.seed(10)
R2_2=m_hast_ej1(250_000,30,r1_30,r2_30,0.0005,0.5,'propuesta2') #Ejecutamos 250 000 iteraciones. Avanzamos con propuesta uniforme en [-0.03,0.03]x[-0.5,0.5]


logf_2_2=np.log(post_ej1graf(30,r1_30,r2_30,R2_2[:20_000,0],R2_2[:20_000,1]))     
plt.plot(logf_2_2,color='r',alpha=0.9,linewidth=0.5)
plt.title('Gráfica Log-densidad vs número de iteraciones, n=30. Propuesta 2')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                                                         #Determinamos que lo mejor es utilizar un burn-in=7000

plt.plot(R2_2[:,0],R2_2[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.3)      #Graficamos el recorrido de la cadena, y las curvas de nivel
plt.title('Algoritmo Metrópolis-Hastings. Recorrido de la cadena. Propuesta 2: avanzar con uniformes. 250 000 muestras para n=30, seed=10')
plt.xlabel('Muestra en alpha')
plt.ylabel('Muestra en beta')
plt.contour(X_ej1,Y_ej1,Z2,levels=250,cmap='plasma',alpha=0.6,linewidths=0.5)
plt.grid()
plt.show()

plt.hist(R2_2[7000:,0],density=True,bins=50,rwidth=0.95,alpha=0.8,color='b')        #Graficamos los histogramas de la variable alpha
plt.title('Histograma de la variable alpha. Caso n=30')
plt.xlabel('Valor en alpha')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R2_2[7000:,1],density=True,bins=50,rwidth=0.95,alpha=0.8,color='r')        #Graficamos los histogramas de la variable beta
plt.title('Histograma de la variable beta. Caso n=30')
plt.xlabel('Valor en beta')
plt.ylabel('Frecuencia (normalizada)')
plt.show()



""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""




np.random.seed(10)                 #Colocamos la semilla seed=10
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------- Ejercicio 2 ------------------------------------------------------------------------------"""
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

#Buscamos simular de una variable aleatoria Gamma(alpha,1) con propuesta Gamma([alpha],1)
#Elegimos alpha=10*pi ~ 31.415926535
#Dado que ya tenemos la densidad gamma en scipy, basta con implementar la parte del cociente de metrópolis hastings y el correspondiente algoritmo.

""" Creamos una función que nos otorga el cociente de Metropolis-Hastings para este ejercicio 2"""
def mh_quotient_ej2(y,x,alpha):
    x_p=np.max(np.array([np.floor(x),1]))
    y_p=np.max(np.array([np.floor(y),1]))               #Dado que hay posibilidad de que el piso de alpha sea 0, colocamos la condición de que sea 1 cuando alpha esté en (0,1)
    q1=ss.gamma.pdf(y,a=alpha)/ss.gamma.pdf(x,a=alpha)  #El cociente f(y)/f(x)
    q2=ss.gamma.pdf(x,a=np.max(y_p))/ss.gamma.pdf(y,a=np.max(x_p)) #El cociente q(x|y)/q(y|x)
    return(q1*q2)


""" Creamos la función del algoritmo M. Hastings, k iteraciones, propuesta 1: simular de Gamma([alpha],1), propuesta 2: simular de Gamma(X[i],1) (avanzar dependiendo del punto actual)"""
def m_hast_ej2(k,alpha,a='propuesta1'):
        if a=='propuesta1':
            X=np.zeros(k,dtype=float)                       #Matriz donde guardar las muestras
            m=900                                           #El punto inicial será alpha=900
            X[0]=m                                          #El primer punto de la cadena será el punto inicial
            for i in range(k-1):
                y=ss.gamma.rvs(a=np.floor(alpha),size=1)                  #Simulamos una propuesta 'y' de la gamma([alpha],1)
                quot=mh_quotient_ej2(y[0],np.floor(alpha),alpha)          #Calculamos el cociente de Metrópolis-Hastings
                U=ss.uniform.rvs(0,1,size=1)                #Generamos una uniforme en (0,1)

                if U<=quot:
                    X[i+1]=y                                #Si el cociente es más grande que la uniforme, aceptamos y nos movemos a 'y'
                else:
                    X[i+1]=X[i]                             #Si el cociente es más chico que la uniforme, rechazamos y nos quedamos en 'x'
            return(X)

        elif a=='propuesta2':
            X=np.zeros(k,dtype=float)                       #Matriz donde guardar las muestras
            m=900                                           #El inicial es alpha=900
            X[0]=m                                          #El primer punto de la cadena será el punto inicial
            for i in range(k-1):
                x=np.max(np.array([np.floor(X[i]),1],dtype=float))
                y=ss.gamma.rvs(a=x,size=1)                  #Simulo una propuesta 'y' de la gamma (que depende del punto inicial).
                quot=mh_quotient_ej2(y[0],x,alpha)          #Calculamos el cociente de Metrópolis-Hastings
                U=ss.uniform.rvs(0,1,size=1)                #Generamos una uniforme en (0,1)

                if U<=quot:
                    X[i+1]=y                                #Si el cociente es más grande que la uniforme, aceptamos y nos movemos a 'y'
                else:
                    X[i+1]=X[i]                             #Si el cociente es más chico que la uniforme, rechazamos y nos quedamos en 'x'
            return(X)

"""---------------------------------------------------------------------- Ejemplo 1, propuesta 1 ---------------------------------------------------------"""

#Propuesta Gamma con parámetro de forma igual a la parte entera de alpha. Elegimos alpha=10*pi

k_1=150_000
R_ej2_1=m_hast_ej2(k_1,10*np.pi,'propuesta1') #Ejecutamos el algoritmo. 150 000 iteraciones

x_ej2=np.linspace(1,k_1,k_1)                    #Rejilla para graficar

logf_ej2_1=np.log(ss.gamma.pdf(R_ej2_1,a=10*np.pi))     
plt.plot(logf_ej2_1[:10000],color='r',alpha=0.9,linewidth=0.5)
plt.title('Gráfica Log-densidad vs número de iteraciones. Propuesta 1') #Graficamos la log-densidad evaluada en el muestreo
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()                                                          #No parece haber un patrón al inicio. Tomamos un burn-in genérico de 1000
plt.show()   


plt.plot(x_ej2,R_ej2_1,linewidth=0.3,color='blue',marker='.',markersize=0.3,alpha=0.8)
plt.plot(np.linspace(0,k_1,k_1),10*np.pi*np.ones(k_1),linewidth=0.5,color='red')            #Visualizamos el recorrido de la caminata.
plt.title('Recorrido de la cadena. Propuesta 1. 150 000 iteraciones, seed=10') #Graficamos la log-densidad evaluada en el muestreo
plt.xlabel('Número de iteración')
plt.ylabel('Valor de x')
plt.show()

plt.hist(R_ej2_1[1000:],density=True,bins=50,rwidth=0.95,color='red')                        #Visualizamos el histograma de alpha.
plt.title('Histograma de la variable. Propuesta 1')
plt.xlabel('Valor muestreado')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

np.mean(R_ej2_1[1000:])                                                                      #Extraemos la media. Esperamos obtener una media de alpha/1~31.415926535


"""---------------------------------------------------------------------- Ejemplo 2, propuesta 2 ---------------------------------------------------------"""
#Una propuesta que depende del punto actual. Elegimos como propuesta una Gamma con parámetro de forma la parte entera del punto actual. Elegimos alpha=10*pi 

k_2=150_000
R_ej2_2=m_hast_ej2(k_2,10*np.pi,'propuesta2') #Ejecutamos el algoritmo. 150 000 iteraciones
x_ej2=np.linspace(1,k_2,k_2)                  #Rejilla para graficar.

logf_ej2_2=np.log(ss.gamma.pdf(R_ej2_2,a=10*np.pi))     
plt.plot(logf_ej2_2[:1000],color='r',alpha=0.9,linewidth=0.5)              #Graficamos la log-densidad
plt.title('Gráfica Log-densidad vs número de iteraciones')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                                  #Observamos un comportamiento anómalo al inicio. Tomamos un burn-in=1000

plt.plot(x_ej2,R_ej2_2,linewidth=0.3,color='blue',marker='.',markersize=0.3,alpha=0.8)
plt.plot(np.linspace(0,k_2,k_2),10*np.pi*np.ones(k_2),linewidth=0.5,color='red')            #Visualizamos el recorrido de la cadena
plt.title('Recorrido de la cadena. Propuesta 1. 150 000 iteraciones, seed=10') #Graficamos la log-densidad evaluada en el muestreo
plt.xlabel('Número de iteración')
plt.ylabel('Valor de x')
plt.show()

plt.hist(R_ej2_2[1000:],density=True,bins=50,rwidth=0.95,color='red')                        #Visualizamos el histograma
plt.title('Histograma de la variable. Propuesta 2')
plt.xlabel('Valor muestreado')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

np.mean(R_ej2_2[1000:])                                                             #Extraemos la media. Esperamos obtener una media de alpha/1~31.415926535


""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""
""" ------------------------------------------------------------------------------------------------------------------------------------------------------"""



np.random.seed(10)                 #Colocamos la semilla seed=10
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------- Ejercicio 3 ------------------------------------------------------------------------------"""
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

#Construimos la función que nos devuelve la distribución objetivo del ejercicio 3 (normal bivariada de media (3,5) y matriz de covarianzas [[1,0.9], [0,9,1]] )  
def dist_obj(x):
    mean=np.array([3,5],dtype=float)
    cov_matrix=np.array([[1,0.9],[0.9,1]],dtype=float)
    f=ss.multivariate_normal.pdf(x,mean=mean,cov=cov_matrix)
    return(f)


#Construimos la función que nos devuelve el cociente de Metrópolis-Hastings del ejercicio 3
def mh_quotient_ej3(y,x):
    q1=dist_obj(y)/dist_obj(x)  #El cociente f(y)/f(x)
    #La propuesta es simétrica, así que la parte de la derecha se anula. Funciona tanto para la propuesta 1 como para la propuesta 2
    return(q1)


""" Creamos la función del algoritmo M. Hastings, k iteraciones. Propuesta 1: avanzar según normales bivariadas. Propuesta 2: avanzar según t de students con 2 grados de libertad"""
def m_hast_ej3(k,m,sigma,a='prop1'):
        if a=='prop1':
            X=np.zeros((k,2),dtype=float)                   #Matriz donde guardar las muestras
            X[0]=m                                          #El primer punto de la cadena será el punto inicial, que denotamos por m
            
            for i in range(k-1):
                y=ss.multivariate_normal.rvs(mean=X[i],cov=sigma*np.identity(2))          #Simulo una propuesta 'y' de la normal multivariada.
                quot=mh_quotient_ej3(y,X[i])                #Calculamos el cociente de Metrópolis-Hastings
                U=ss.uniform.rvs(0,1,size=1)                #Generamos una uniforme en (0,1)

                if U<=quot:
                    X[i+1]=y                                #Si el cociente es más grande que la uniforme, aceptamos y nos movemos a 'y'
                else:
                    X[i+1]=X[i]                             #Si el cociente es más chico que la uniforme, rechazamos y nos quedamos en 'x'
            return(X)

        elif a=='prop2':
            
            X=np.zeros((k,2),dtype=float)                   
            X[0]=m         

            for i in range(k-1):
                y=ss.multivariate_t.rvs(loc=X[i],shape=sigma*np.identity(2),df=2)               #Simulo una propuesta de la t de student multivariada
                quot=mh_quotient_ej3(y,X[i])             #Calculamos el cociente de Metrópolis-Hastings
                U=ss.uniform.rvs(0,1,size=1)             #Generamos una uniforme en (0,1)

                if U<=quot:
                    X[i+1]=y                                #Si el cociente es más grande que la uniforme, aceptamos y nos movemos a 'y'
                else:
                    X[i+1]=X[i]                             #Si el cociente es más chico que la uniforme, rechazamos y nos quedamos en 'x'
            return(X)

""" --------------------------------------Ejemplo 1. Propuesta 1: avanzamos según una normal multivariada con media 0 y varianza sigma*I_2--------------------------------"""
m_1=np.array([35,35],dtype=float)
R3_1=m_hast_ej3(150_000,m_1,0.5,'prop1')                            #Para ver cómo funciona el algoritmo, comenzamos en el punto (35,35). Sigma lo escogemos igual a 0.5

x_ej3=np.linspace(-2, 40, 300)
y_ej3=np.linspace(-2, 40, 300)                                      #Generamos rejillas para graficar las curvas de nivel de la objetivo
X3,Y3=np.meshgrid(x_ej3,y_ej3)
data= np.dstack((X3,Y3))
Z3=dist_obj(data)

logf_ej3_1=np.log(dist_obj(R3_1))     
plt.plot(logf_ej3_1[:2000],color='r',alpha=0.9,linewidth=0.5)                           #Graficamos la log-densidad objetivo evaluada en nuestra muestra
plt.title('Gráfica Log-densidad vs número de iteraciones. Propuesta normal bivariada')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                                                         #Determinamos que lo mejor es utilizar un burn-in=2000

plt.plot(R3_1[:,0],R3_1[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.3)  #Graficamos el recorrido de la cadena y las curvas de nivel de la densidad objetivo
plt.title('Algoritmo Metrópolis-Hastings. Recorrido de la cadena. Propuesta 1: normales bivariadas N(0,0.5*I_2). 150 000 muestras, seed=10')
plt.xlabel('Muestreo en x')
plt.ylabel('Muestreo en y')
plt.contour(X3,Y3,Z3,levels=100,cmap='plasma',alpha=0.5,linewidths=0.5)
plt.grid()
plt.show()

plt.hist(R3_1[2000:,0],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        #Graficamos los histogramas de la variable x
plt.title('Histograma de la variable x. Propuesta normal')
plt.xlabel('Valor en x')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R3_1[2000:,1],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        #Graficamos los histogramas de la variable y
plt.title('Histograma de la variable y. Propuesta normal')
plt.xlabel('Valor en y')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

""" --------------------------------------Ejemplo 2. Propuesta 2: avanzamos según una t-student con 2 grados de libertad en ambas variables--------------------------------"""

m_1=np.array([35,35],dtype=float)
R3_2=m_hast_ej3(150_000,m_1,0.3,'prop2')                            #Para ver cómo funciona el algoritmo, comenzamos en el punto (35,35). Sigma lo escogemos igual a 0.3

logf_ej3_2=np.log(dist_obj(R3_2))                                   #Graficamos la log-densidad objetivo evaluada en nuestra muestra
plt.plot(logf_ej3_2[:2000],color='r',alpha=0.9,linewidth=0.5)
plt.title('Gráfica Log-densidad vs número de iteraciones. Propuesta t-student 2 df')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                                                         #Determinamos que lo mejor es utilizar un burn-in=2000

plt.plot(R3_2[:,0],R3_2[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.3)    #Graficamos el recorrido de la cadena y las curvas de nivel de la densidad objetivo
plt.title('Algoritmo Metrópolis-Hastings. Recorrido de la cadena. Propuesta 2: t-student 2-variada con 2 df, scale= 0.3*I_2. 150 000 muestras, seed=10')
plt.xlabel('Muestreo en x')
plt.ylabel('Muestreo en y')
plt.contour(X3,Y3,Z3,levels=100,cmap='plasma',alpha=0.5,linewidths=0.5)
plt.grid()
plt.show()

plt.hist(R3_2[2000:,0],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        #Graficamos los histogramas de la variable x
plt.title('Histograma de la variable x. Propuesta t-student 2df')
plt.xlabel('Valor en x')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R3_2[2000:,1],density=True,bins=50,rwidth=0.95,alpha=0.7,color= 'red')        #Graficamos los histogramas de la variable y
plt.title('Histograma de la variable y. Propuesta t-student 2df')
plt.xlabel('Valor en y')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

""" --------------------------------------Experimento. Propuesta 1: avanzamos según una normal bivariada. Punto inicial (1000,1)--------------------------------"""

m_2=np.array([1000,1],dtype=float)
R3_3=m_hast_ej3(10_000,m_2,0.5,'prop1')                #Con un punto inicial demasiado alejado, la cadena no puede salir del mismo.
R3_3            





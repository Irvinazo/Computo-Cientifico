import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.special import gamma


""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------- Ejercicio 1 ------------------------------------------------------------------------------"""
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

np.random.seed(10)                 #Colocamos la semilla seed=10 para este problema.

""" Creamos una función que calcula la distribución objetivo. En este caso, una normal bivariada de media (0,0), sigma1=sigma2=1, y rho=0.8 o rho=0.95"""
def dist_obj(x,rho):
     mean=np.array([0,0])
     cov_mat=np.array([[1,rho],[rho,1]],dtype=float)    
     f=ss.multivariate_normal.pdf(x,mean=mean,cov=cov_mat)
     return(f)

""" Creamos la función del algoritmo M. Hastings con Kernel Híbrido y Gibbs Sampler, k iteraciones, vector inicial m y correlación rho"""
def MH_gibbs1(k,rho,m):
        X=np.zeros((k,2),dtype=float)                   #Matriz donde guardar las muestras
        X[0]=m                                          #Mi primer punto de la cadena será m
        for i in range(k-1):
            y=np.array([0,0],dtype=float)               #Creamos un vector vacío para guardar la propuesta
            uni=ss.uniform.rvs(0,1)                     #'Tiramos una moneda justa'
            if uni<=0.5:                                #Si la moneda cae sol, elegimos el primer kernel
                y1=ss.norm.rvs(rho*X[i,1], np.sqrt(1-rho**2))          #Simulamos una propuesta 'y1' de la normal de la propuesta 1
                y[0]=y1
                y[1]=X[i,1]                                   #Fijamos el eje Y, y la propuesta la colocamos en el eje X
                X[i+1]=y                                #Dado que estamos utilizando un Gibbs Sampler (muestreamos de condicionales 
                                                        #de un elemento del vector de parámetros dados los demás), aceptamos siempre
            else:
                y2=ss.norm.rvs(rho*X[i,0], np.sqrt(1-rho**2))    #Si la moneda cae águila, elegimos el segundo kernel.
                y[0]=X[i,0]                             #Simulamos una propuesta 'y2' de la normal de la propuesta 2
                y[1]=y2                                 #Fijamos el eje X, y la propuesta la colocamos en el eje Y
                X[i+1]=y                                #Mismo argumento: la probabilidad de aceptar siempre es 1
        return(X)


m=np.array([5,5],dtype=float)                         #Establecemos el punto (5,5) como el punto inicial  

""" --------------------------------------Ejemplo 1 rho=0.8--------------------------------"""
rho1=0.8                                               #Establecemos en el primer ejemplo rho=0.8
R1=MH_gibbs1(250_000,rho1,m)                           #Ejecutamos el algoritmo con 250 000 iteraciones, rho1=0.8, y el vector inicial (5,5)

x_ej1=np.linspace(-5, 5, 100)
y_ej1=np.linspace(-5, 5, 100)                        #Generamos rejillas para graficar las curvas de nivel de la objetivo
X1,Y1=np.meshgrid(x_ej1,y_ej1)
data= np.dstack((X1,Y1))
Z1_1=dist_obj(data,rho1)

logf_ej1_1=np.log(dist_obj(R1,rho1))     
plt.plot(logf_ej1_1[:5000],color='r',alpha=0.9,linewidth=0.5)        #Graficamos la log-densidad objetivo evaluada en nuestra muestra
plt.title(r'Gráfica Log-densidad vs número de iteraciones.    Coeficiente de correlación $\rho=0.8$')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                  

#Determinamos que lo mejor es utilizar un burn-in=250

plt.plot(R1[:,0],R1[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.2)  #Graficamos el recorrido de la cadena y las curvas de nivel de la densidad objetivo
plt.contour(X1,Y1,Z1_1,levels=100,cmap='plasma',alpha=0.6,linewidths=0.4)
plt.title(r'Algoritmo MH con Kerneles Híbridos.    Recorrido de la cadena.    Coeficiente de correlación $\rho=0.8$.    250 000 muestras.   $seed$=10')
plt.xlabel('Muestreo en $x$')
plt.ylabel('Muestreo en $y$')
plt.grid()
plt.show()

plt.hist(R1[250:,0],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        #Graficamos los histogramas de la variable x
plt.title(r'Histograma de la variable $x$.  Coeficiente de correlación $\rho=0.8$')
plt.plot(x_ej1,ss.norm.pdf(x_ej1), color='red', alpha=0.8)
plt.xlabel('Valor en $x$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R1[250:,1],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        #Graficamos los histogramas de la variable y
plt.title(r'Histograma de la variable $y$.  Coeficiente de correlación $\rho=0.8$')
plt.plot(x_ej1,ss.norm.pdf(x_ej1), color='blue', alpha=0.8)
plt.xlabel('Valor en $y$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()


""" --------------------------------------Ejemplo 2 rho=0.95--------------------------------"""
rho2=0.95                                              #Establecemos en el segundo ejemplo rho=0.95
R2=MH_gibbs1(250_000,rho2,m)                           #Ejecutamos el algoritmo con 250 000 iteraciones, rho2=0.95, y el vector inicial (5,5)
Z1_2=dist_obj(data,rho2)


logf_ej1_2=np.log(dist_obj(R2,rho2))     
plt.plot(logf_ej1_2[:2000],color='r',alpha=0.9,linewidth=0.5)                           #Graficamos la log-densidad objetivo evaluada en nuestra muestra
plt.title(r'Gráfica Log-densidad vs número de iteraciones.    Coeficiente de correlación $\rho=0.95$')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                  

#Determinamos que lo mejor es utilizar un burn-in=250

plt.plot(R2[:,0],R2[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.2)  #Graficamos el recorrido de la cadena y las curvas de nivel de la densidad objetivo
plt.contour(X1,Y1,Z1_2,levels=100,cmap='plasma',alpha=0.6,linewidths=0.4)
plt.title(r'Algoritmo MH con Kerneles Híbridos.    Recorrido de la cadena.   Coeficiente de correlación $\rho=0.95$.   250 000 muestras.   $seed$=10')
plt.xlabel(r'Muestreo en $x$')
plt.ylabel(r'Muestreo en $y$')
plt.grid()
plt.show()

plt.hist(R2[250:,0],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        #Graficamos los histogramas de la variable x
plt.title(r'Histograma de la variable $x$.   Coeficiente de correlación $\rho=0.95$')
plt.plot(x_ej1,ss.norm.pdf(x_ej1), color='red', alpha=0.8)
plt.xlabel(r'Valor en $x$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R2[250:,1],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        #Graficamos los histogramas de la variable y
plt.title(r'Histograma de la variable $y$.    Coeficiente de correlación $\rho=0.95$')
plt.plot(x_ej1,ss.norm.pdf(x_ej1), color='blue', alpha=0.8)
plt.xlabel(r'Valor en $y$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()





""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------- Ejercicio 2 ------------------------------------------------------------------------------"""
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

np.random.seed(20)                 #Colocamos la semilla seed=20

"""Creamos una función que calcula la distribución objetivo"""
def dist_obj2(data,alpha,lamb):
    if alpha>0:
        if lamb>0:                                      #Imponemos estas condiciones para evitar que los parámetros se salgan del soporte
            suma=np.sum(data**alpha)
            producto=np.prod(data)
            p1=np.exp(-lamb*(1+suma))
            p2=np.exp(-alpha)*(1+producto)**alpha
            p3=(alpha**20)*(lamb**(19+alpha))/gamma(alpha)
            f=p1*p2*p3
            return(f)
        else:
            f=0
        return(f)
    else:
        f=0
    return(f)
    
"""Creamos una función que calcula la propuesta 1:"""
def prop1_lambda(data,alpha):
    suma=np.sum(data**alpha)
    return(ss.gamma.rvs(a=alpha+20,scale=1/(1+suma)))

"""Creamos una función que calcula la propuesta 2:"""
def prop2_lambda(data):
    suma=np.sum(np.log(data))
    if 1-suma>0:                    #Dado que esta cantidad es sensible, imponemos esta condición si su valor se vuelve negativo
        return(ss.gamma.rvs(21,scale=1/(1-suma)))
    else:
        return(-1)

"""Creamos una función que calcula la propuesta 3:"""
def prop3(c=1,b=1):
    alpha=ss.expon.rvs(c)
    lambda_alpha=ss.gamma.rvs(alpha,scale=1/b)
    return(alpha,lambda_alpha)

"""Creamos una función que calcula la propuesta 4:"""
def prop4_alpha(alpha,sigma=0.5):
    eps=ss.norm.rvs(0,sigma)
    alpha_p=alpha+eps
    return(alpha_p)

"""Creamos el algoritmo de MH - Kerneles Híbridos y Gibbs Sampler. k iteraciones, vect. inicial m= a priori alpha~exp(1), lambda|alpha~Gamma(alpha,1). 4 Propuestas"""

def MH_gibbs2(data,k,m):
    X=np.zeros((k,2),dtype=float)                   #Matriz donde guardar las muestras
    X[0]=m                                          #Mi primer punto de la cadena será m
    for i in range(k-1):
        y=np.array([0,0],dtype=float)               #Creamos un vector vacío para guardar la propuesta
        uni=ss.uniform.rvs(loc=0,scale=1)           #Lanzamos una uniforme

        if uni<=0.25:                               #Si la uniforme está en el primer cuarto del intervalo [0,1]
            y1=prop1_lambda(data,X[i,0])            #Simulamos lambda de la propuesta 1
            y[1]=y1                                 #Proponemos reemplazar el valor de lambda actual por esta nueva propuesta
            y[0]=X[i,0]                             #Pero mantenemos el valor de alpha

            #Dado que se trata de un Kernel Gibbs en este caso, la propuesta siempre se acepta y directamente hacemos
            X[i+1]=y

        elif 0.25<uni<=0.5:                         #Si la uniforme está en el segundo cuarto del intervalo
            y2=prop2_lambda(data)                   #Simulamos lambda de la propuesta 2
            if y2==-1:
                X[i+1]=X[i]
            else:
                y[1]=X[i,1]                             #Mantenemos lambda en su lugar
                y[0]=y2                                 #Proponemos reemplazar el alpha actual por el simulado
                quot=dist_obj2(data,y[0],y[1])/dist_obj2(data,X[i,0],X[i,1]) #Calculamos el cociente de M-Hastings. No calculamos el cociente de las propuestas
                U=ss.uniform.rvs(0,1)
                if U<quot:
                    X[i+1]=y                                    
                else:
                    X[i+1]=X[i]

        elif 0.5<uni<=0.75:                         #Si la uniforme está en el tercer cuarto
            y=prop3()                               #Simulamos tanto alpha como lambda de la propuesta 3 y reemplazamos los valores de alpha y lambda por estos
            qxy_qyx=np.exp(y[0]+y[1]-X[i,0]-X[i,1])*(X[i,1]**(X[i,0]-1))*(y[1]**(1-y[0]))*(gamma(y[0]))/(gamma(X[i,0])) #Calculamos el cociente q((a,la)|(a_p,la_p))/q((a_p,la_p)|(a,la))
            quot1=dist_obj2(data,y[0],y[1])/dist_obj2(data,X[i,0],X[i,1])
            quot=quot1*qxy_qyx
            U=ss.uniform.rvs(0,1)
            if U<quot:
                 X[i+1]=y
            else:
                X[i+1]=X[i]

        else:                                       #Si la uniforme está en el último cuarto
            y4=prop4_alpha(X[i,0])                  #Muestreamos alpha de acuerdo a la propuesta 4
            y[0]=y4                                 #Proponemos este nuevo alpha 
            y[1]=X[i,1]                             #Pero mantenemos el lambda actual

            #Aquí la propuesta es simétrica, de tal forma que el cociente de metrópolis está dado simplemente por
            quot=dist_obj2(data,y[0],y[1])/dist_obj2(data,X[i,0],X[i,1])
            U=ss.uniform.rvs(0,1)
            if U<quot:
                X[i+1]=y
            else:
                X[i+1]=X[i]
    return(X)


"""Simulamos datos de una variable aleatoria Weibull(1,1). 20 datos"""
weib=ss.weibull_min.rvs(c=1, loc=0, scale=1, size=20)

"""Hacemos el vector inicial m"""
m=np.array([2,2],dtype=float)

#Realizamos MHhastings, 250_000 iteraciones, vector inicial m=(2,2)
k=250_000
R3=MH_gibbs2(weib,k,m)
R3

""" --------------------------------------Ejemplo 1 --------------------------------"""                                              #Establecemos en el primer ejemplo rho=0.8

x_ej2=np.linspace(0, 3, 100)
y_ej2=np.linspace(0, 3, 100)                        #Generamos rejillas para graficar las curvas de nivel de la objetivo
X2,Y2=np.meshgrid(x_ej2,y_ej2)
data2= np.dstack((X2,Y2))
Z2=np.zeros((100,100),dtype=float)
for i in range(100):
    for j in range(100):
        Z2[i,j]=dist_obj2(weib,data2[i][j][0],data2[i][j][1])

logf_ej2=np.zeros(k,dtype=float)
for i in range(k):
    logf_ej2[i]=np.log(dist_obj2(weib,R3[i,0],R3[i,1]))     

plt.plot(logf_ej2[:2000],color='r',alpha=0.9,linewidth=0.5)                           #Graficamos la log-densidad objetivo evaluada en nuestra muestra
plt.title('Gráfica Log-densidad vs número de iteraciones.')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                  

#Determinamos que lo mejor es utilizar un burn-in=250

plt.plot(R3[:,0],R3[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.2)  #Graficamos el recorrido de la cadena y las curvas de nivel de la densidad objetivo
plt.contour(X2,Y2,Z2,levels=100,cmap='plasma',alpha=0.6,linewidths=0.3)
plt.title(r'Algoritmo MH con Kerneles Híbridos. Vector inicial $(\alpha,\lambda)=(2,2)$')
plt.xlabel(r'Muestreo en $\alpha$')
plt.ylabel(r'Muestreo en $\lambda$')
plt.grid()
plt.show()

plt.hist(R3[250:,0],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        #Graficamos los histogramas de la variable alpha
plt.title(r'Histograma de la variable $\alpha$.')
plt.xlabel(r'Valor en $\alpha$')
plt.ylabel(r'Frecuencia (normalizada)')
plt.show()

plt.hist(R3[250:,1],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        #Graficamos los histogramas de la variable lambda
plt.title(r'Histograma de la variable $\lambda$.')
plt.xlabel(r'Valor en $\lambda$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()



""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------- Ejercicio 3 ------------------------------------------------------------------------------"""
""" ----------------------------------------------------------------------------------------------------------------------------------------------------------------------"""

#Colocamos la semilla seed=10

np.random.seed(10)

"""Construimos una función para la propuesta 1 (consistente en realidad en 10 propuestas)"""
def propuesta1(table,alpha,beta,i):
    lambdai=ss.gamma.rvs(a=table[1,i]+alpha,scale=1/(beta+table[0,i]))
    return(lambdai)

"""Construimos una función para la propuesta 2"""
def propuesta2(alpha,gama,delta,data):
    beta=ss.gamma.rvs(a=10*alpha+gama,scale=1/(delta+sum(data)))
    return(beta)

#Cargamos los datos
tabla=np.array([[94.32,15.72,62.88,125.76,5.24,31.44,1.05,1.05,2.1,10.48],[5,1,5,14,3,17,1,1,4,22]],dtype=float)

m=np.array([5,5,5,5,5,5,5,5,5,5,5],dtype=float) #Agregamos un punto inicial consistente en un vector de puros 5.

"""Construimos la función del algoritmo MH para este ejercicio. La estructura de esta función es similar a las anteriores."""
def MH_gibbs3(k,tabla,alpha,gama,delta):
    X=np.zeros((k,11),dtype=float)                   
    X[0]=m   
    d=1-1/11                                        #Dividimos el intervalo [0,1] en 11 pedazos iguales
    for i in range(k-1):
        uni=ss.uniform.rvs(loc=0,scale=1)           #Lanzamos una uniforme. Dependiendo del valor de la misma, usaremos cierto kernel
        if uni<=d:
            j=ss.randint.rvs(0,10)                          #Si estamos en alguno de los primeros 10 pedazos del intervalo, simularemos para las lambda's
            y1=propuesta1(tabla,alpha,X[i,10],j)            
            X[i+1]=X[i]
            X[i+1,j]=y1                                     #Dado que estamos trabajando con Gibbs Sampler (ver reporte) la probabilidad de aceptar es 1.

        else:                                               #Si la uniforme cae en el último pedazo del intervalo, simularemos para la beta.
            y2=propuesta2(alpha,gama,delta,X[i,:10])        #Simulamos beta de la propuesta 2
            X[i+1]=X[i]
            X[i+1,10]=y2                                    #Mismo razonamiento que antes con la aceptación de la propuesta
    return(X)

#Establecemos los parámetros pedidos:
alpha=1.8
gama=0.01
delta=1
beta=5
k3=250_000

#Implementamos MH-con Kerneles Híbridos y Gibbs sampler
R4=MH_gibbs3(k3,tabla,alpha,gama,delta)

#Basados meramente en la heurística de los anteriores resultados, utilizamos un burn-in de 250. Esto se hace sin revisar la log-densidad objetivo.
plt.hist(R4[250:,0],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        #Graficamos los histogramas de cada una de las 11 variables.
plt.title(r'Histograma de la variable $\lambda_1$.')
plt.xlabel(r'Valor en $\lambda_1$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R4[250:,1],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')       
plt.title(r'Histograma de la variable $\lambda_2$.')
plt.xlabel(r'Valor en $\lambda_2$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()


plt.hist(R4[250:,2],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        
plt.title(r'Histograma de la variable $\lambda_3$.')
plt.xlabel(r'Valor en $\lambda_3$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R4[250:,3],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        
plt.title(r'Histograma de la variable $\lambda_4$.')
plt.xlabel(r'Valor en $\lambda_4$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()


plt.hist(R4[250:,4],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        
plt.title(r'Histograma de la variable $\lambda_5$.')
plt.xlabel(r'Valor en $\lambda_5$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R4[250:,5],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        
plt.title(r'Histograma de la variable $\lambda_6$.')
plt.xlabel(r'Valor en $\lambda_6$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()


plt.hist(R4[250:,6],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        
plt.title(r'Histograma de la variable $\lambda_7$.')
plt.xlabel(r'Valor en $\lambda_7$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R4[250:,7],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        
plt.title(r'Histograma de la variable $\lambda_8$.')
plt.xlabel(r'Valor en $\lambda_8$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()


plt.hist(R4[250:,8],density=True,bins=50,rwidth=0.95,alpha=0.7,color='b')        
plt.title(r'Histograma de la variable $\lambda_9$.')
plt.xlabel(r'Valor en $\lambda_9$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

plt.hist(R4[250:,9],density=True,bins=50,rwidth=0.95,alpha=0.7,color='r')        
plt.title(r'Histograma de la variable $\lambda_{10}$.')
plt.xlabel(r'Valor en $\lambda_{10}$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()


plt.hist(R4[250:,10],density=True,bins=50,rwidth=0.95,alpha=0.7,color='g')        
plt.title(r'Histograma de la variable $\beta$.')
plt.xlabel(r'Valor en $\beta$')
plt.ylabel('Frecuencia (normalizada)')
plt.show()

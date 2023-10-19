import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt 


"""----------------- Ejercicio 1: simular 5 y 40 v.a. Bernoulli Be(1/3) respectivamente. Contar el número de éxitos--------------------"""


np.random.seed(10)                      #Colocamos semilla para replicar resultados
n5=ss.bernoulli.rvs(1/3,size=5)
n40=ss.bernoulli.rvs(1/3,size=40)       #Creamos las muestras

r5=np.sum(n5)
r40=np.sum(n40)                         #Obtenemos el número de éxitos respectivos. Con la semilla anterior, r5=2, r40=13



"""------------------ Ejercicio 2: implementar MH para simular de la posterior dada. La propuesta (p'|p)=p' ~ Beta(r+1,n-r+1), la dist. inicial m ~ U(0,1/2)--------------------"""



""" Creamos la función que calcula la densidad de la posterior evaluada en un punto p"""
def post(p,n,r):
    
    if p<=(1/2):                                #El condicional emula la indicadora
        f=(p**r)*((1-p)**(n-r))*np.cos(np.pi*p)
    else:
        f=0
    return(f)

""" Creamos la función que calcula el cociente de Metrópolis-Hastings con punto actual 'x' y punto futuro 'y' de la propuesta y densidad objetivo del problema"""
def mh_quotient(y,x,n,r):
    q1=(post(y,n,r)/post(x,n,r))                            #El cociente f(y)/f(x)
    q2=ss.beta.pdf(x,r+1,n-r+1)/ss.beta.pdf(y,r+1,n-r+1)    #El cociente q(x|y)/q(y|x)
    return(q1*q2)

""" Creamos una función para el algoritmo de Metrópolis-Hastings de este ejercicio, con k iteraciones, densidad beta de parámetros n y r"""
def m_hast_beta(k,n,r):
         
        X=np.zeros(k,dtype=float)                       #Vector donde guardar las muestras
        m=ss.uniform.rvs(0,1/2,size=1)                  #Simulo un punto en la distribución inicial.
        X[0]=m                                          #Mi primer punto de la cadena será el punto inicial

        for i in range(k-1):
            y=ss.beta.rvs(r+1,n-r+1,size=1)             #Simulo una propuesta 'y' de la beta.
            quot=mh_quotient(y,X[i],n,r)                #Calculamos el cociente de Metrópolis-Hastings
            U=ss.uniform.rvs(0,1,size=1)                #Generamos una uniforme en (0,1)

            if U<=quot:
                X[i+1]=y                                #Si el cociente es más grande que la uniforme, aceptamos y nos movemos a 'y'
            else:
                X[i+1]=X[i]                             #Si el cociente es más chico que la uniforme, rechazamos y nos quedamos en 'x'
        return(X)


#------------------------------- Ejemplo 1: n=5, r5=2 ----------------------

mh_1=m_hast_beta(1000000,5,r5)               #Ejecutamos Metropolis-Hastings 1 millón de veces
x=np.linspace(0,1/2,100)                     #Creamos una rejilla para graficar

y1=np.zeros(100)                             #Creamos un vector con los valores de la distribución objetivo evaluados en la rejilla
for i in range(100):
     y1[i]=post(x[i],5,r5)

logf_1=np.zeros(len(mh_1))                    #Creamos también un vector con los valores de la log-densidad evaluada en la rejilla
for i in range(len(mh_1)):
     logf_1[i]=np.log(post(mh_1[i],5,r5))     #La idea es utilizar esta gráfica para el burn-in
     
plt.plot(logf_1)                                  #Graficamos la log-densidad evaluada en las muestras para ver el comportamiento de las mismas
plt.title('Ejemplo 1: log-densidad objetivo evaluada en las muestras')
plt.xlabel('Número de iteración')
plt.ylabel('Log-valor de la muestra')
plt.show()                                                #Seleccionamos un burn-in de 50 000 basado en la gráfica

plt.plot(x,y1*188.04733,color='blue')                     #Utilizando Wolfram, hallamos que la constante de normalización es la constante que multiplica a 'y' aquí
plt.hist(mh_1[50_000:],density=True,bins=50,rwidth=0.95,color='orange')
plt.title('Metrópolis-Hastings, 1 millón de muestras para n=5, r=2, seed=10, burn-in=50 000')
plt.xlabel('Valores de la muestra')
plt.ylabel('Frecuencia (normalizada)')
plt.legend(('Densidad objetivo f','Histograma de muestras'))
plt.show()


#------------------------------- Ejemplo 2: n=40, r40=13 ---------------------

mh_2=m_hast_beta(1000000,40,r40)                    #Ejecutamos Metropolis-Hastings 1 millón de veces

y2=np.zeros(100)                                    #Creamos un vector con los valores de la distribución objetivo evaluados en la rejilla
for i in range(100):
     y2[i]=post(x[i],40,r40)

logf_2=np.zeros(len(mh_2))                          #Creamos el vector con los valores de la log-densidad evaluada en la rejilla
for i in range(len(mh_2)):
     logf_2[i]=np.log(post(mh_2[i],40,r40))         
     
plt.plot(logf_2)                                              #Usando la gráfica, decidimos hacer un Burn-in hasta el valor 50 000
plt.title('Ejemplo 2: log-densidad objetivo evaluada en las muestras')
plt.xlabel('Número de iteración')
plt.ylabel('Log-valor de la muestra')
plt.show()

plt.plot(x,y2*1009076644417,color='blue')                     #Utilizando wolfram, hallamos que la constante de normalización es la constante que multiplica a 'y' aquí
plt.hist(mh_2[800_000:],density=True,bins=50,rwidth=0.95,color='orange')
plt.title('Metrópolis-Hastings, 1 millón de muestras para n=40, r=13, seed=10, burn-in=50 000')
plt.xlabel('Valores de la muestra')
plt.ylabel('Frecuencia (normalizada)')
plt.legend(('Densidad objetivo f','Histograma de muestras'))
plt.show()



"""------------------ Ejercicio 5: implementar MH para simular de la posterior dada. Usamos la propuesta (p'|p)=p'~Unif(0,1), la dist. inicial m ~ U(0,1/2)--------------------"""



#Dado que una variable uniforme en $0,1/2$ es una beta(1,1) en (0,1), podemos reutilizar parte del código anterior.

""" Creamos la función que calcula el cociente de Metrópolis-Hastings con punto actual 'x' y punto futuro 'y' de la propuesta y densidad objetivo de este nuevo problema"""
def mh_quotient_2(y,x,n,r):
    q1=(post(y,n,r)/post(x,n,r))                            #El cociente f(y)/f(x)
    q2=ss.beta.pdf(x,1,1)/ss.beta.pdf(y,1,1)    #El cociente q(x|y)/q(y|x)
    return(q1*q2)

""" Creamos una función para el algoritmo de Metrópolis-Hastings de este nuevo ejercicio, con k iteraciones, densidad beta de parámetros 1 y 1"""
def m_hast_beta_2(k,n,r):
         
        X=np.zeros(k,dtype=float)                       #Vector donde guardar las muestras
        m=ss.uniform.rvs(0,1/2,size=1)                  #Simulo un punto en la distribución inicial.
        X[0]=m                                          #Mi primer punto de la cadena será el punto inicial

        for i in range(k-1):
            y=ss.beta.rvs(1,1,size=1)             #Simulo una propuesta 'y' de la beta.
            quot=mh_quotient_2(y,X[i],n,r)                #Calculamos el cociente de Metrópolis-Hastings
            U=ss.uniform.rvs(0,1,size=1)                #Generamos una uniforme en (0,1)

            if U<=quot:
                X[i+1]=y                                #Si el cociente es más grande que la uniforme, aceptamos y nos movemos a 'y'
            else:
                X[i+1]=X[i]                             #Si el cociente es más chico que la uniforme, rechazamos y nos quedamos en 'x'
        return(X)

#------------------------------- Ejemplo 1: n=5, r5=2 ----------------------

mh_3=m_hast_beta_2(1000000,5,r5)               #Ejecutamos Metropolis-Hastings 1 millón de veces

logf_3=np.zeros(len(mh_3))                    #Creamos un vector con los valores de la log-densidad evaluada en la rejilla
for i in range(len(mh_3)):
     logf_3[i]=np.log(post(mh_3[i],5,r5))     #La idea es utilizar esta gráfica para el burn-in
     
plt.plot(logf_3)                                  #Graficamos la log-densidad evaluada en las muestras para ver el comportamiento de las mismas
plt.title('Ejemplo 1: log-densidad objetivo evaluada en las muestras. Propuesta uniforme.')
plt.xlabel('Número de iteración')
plt.ylabel('Log-valor de la muestra')
plt.show()                                                #Seleccionamos un burn-in de 50 000 basado en la gráfica

plt.plot(x,y1*188.04733,color='blue')                     #Utilizando Wolfram, hallamos que la constante de normalización es la constante que multiplica a 'y' aquí
plt.hist(mh_3[50_000:],density=True,bins=50,rwidth=0.95,color='orange')
plt.title('Metrópolis-Hastings, 1 millón de muestras para n=5, r=2, seed=10, burn-in=50 000. Propuesta uniforme.')
plt.xlabel('Valores de la muestra')
plt.ylabel('Frecuencia (normalizada)')
plt.legend(('Densidad objetivo f','Histograma de muestras'))
plt.show()


#------------------------------- Ejemplo 2: n=40, r40=13 ---------------------

mh_4=m_hast_beta_2(1000000,40,r40)                    #Ejecutamos Metropolis-Hastings 1 millón de veces

logf_4=np.zeros(len(mh_4))                          #Creamos el vector con los valores de la log-densidad evaluada en la rejilla
for i in range(len(mh_4)):
     logf_4[i]=np.log(post(mh_4[i],40,r40))         
     
plt.plot(logf_4)                                              #Usando la gráfica, decidimos hacer un Burn-in hasta el valor 50 000
plt.title('Ejemplo 2: log-densidad objetivo evaluada en las muestras. Propuesta uniforme.')
plt.xlabel('Número de iteración')
plt.ylabel('Log-valor de la muestra')
plt.show()

plt.plot(x,y2*1009076644417,color='blue')                     #Utilizando wolfram, hallamos que la constante de normalización es la constante que multiplica a 'y' aquí
plt.hist(mh_4[800_000:],density=True,bins=50,rwidth=0.95,color='orange')
plt.title('Metrópolis-Hastings, 1 millón de muestras para n=40, r=13, seed=10, burn-in=50 000. Propuesta uniforme.')
plt.xlabel('Valores de la muestra')
plt.ylabel('Frecuencia (normalizada)')
plt.legend(('Densidad objetivo f','Histograma de muestras'))
plt.show()


#---------------------------- Contraste entre histogramas de muestreos en el caso n=40, r=13, seed=10, burn-in=20 000, 200 000 muestras -------------------

plt.plot(x,y2*1009076644417,color='blue')                     #Utilizando wolfram, hallamos que la constante de normalización es la constante que multiplica a 'y' aquí
plt.hist(mh_4[20_000:200_000],density=True,bins=50,rwidth=0.95,color='orange')
plt.title('Metrópolis-Hastings, 200 000 muestras para n=40, r=13, seed=10, burn-in=20 000. Propuesta uniforme.')
plt.xlabel('Valores de la muestra')
plt.ylabel('Frecuencia (normalizada)')
plt.legend(('Densidad objetivo f','Histograma de muestras'))
plt.show()

plt.plot(x,y2*1009076644417,color='blue')                     #Utilizando wolfram, hallamos que la constante de normalización es la constante que multiplica a 'y' aquí
plt.hist(mh_2[20_000:200_000],density=True,bins=50,rwidth=0.95,color='orange')
plt.title('Metrópolis-Hastings, 200 000 muestras para n=40, r=13, seed=10, burn-in=20 000. Propuesta beta.')
plt.xlabel('Valores de la muestra')
plt.ylabel('Frecuencia (normalizada)')
plt.legend(('Densidad objetivo f','Histograma de muestras'))
plt.show()
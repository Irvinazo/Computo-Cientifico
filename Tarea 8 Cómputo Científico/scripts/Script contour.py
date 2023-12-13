
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
plt.title('Gráfica Log-densidad vs número de iteraciones. Coef. de correlación rho=0.8')
plt.xlabel('Número de iteración')
plt.ylabel('Log-densidad')
plt.grid()
plt.show()                  

#Determinamos que lo mejor es utilizar un burn-in=250

plt.plot(R3[:,0],R3[:,1],'b',color='gray',linewidth=0.09,alpha=0.9,marker='.', markersize=0.2)  #Graficamos el recorrido de la cadena y las curvas de nivel de la densidad objetivo
plt.contour(X2,Y2,Z2,levels=100,cmap='plasma',alpha=0.6,linewidths=0.3)
plt.title('Algoritmo MH con Kerneles Híbridos. Recorrido de la cadena. Coeficiente de correlación rho=0.8. 250 000 muestras, seed=10')
plt.xlabel('Muestreo en x')
plt.ylabel('Muestreo en y')
plt.grid()
plt.show()
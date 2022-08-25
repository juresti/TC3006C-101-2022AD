from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#Cargar dataset moons
X,y = datasets.make_moons(1,noise=0.15,random_state=303) #Definir cuántos patrones vamos a predecir (Forward)
print("**** Dataset X ***")
print(X) #2 valores de entrada
print("*** Dataset Y ***")
print(y) #1 valor de salida
#plt.scatter(X[:,0],X[:,1],y,marker="o",color="red")
#plt.show()

#*** Definir arquitectura
inputDim = 2
hiddenDim = 1
outputDim = 1

#Pesos input to hidden W1 y bias para cada neurona b1
W1 = np.random.randn(inputDim,hiddenDim) #floats distrubuidos normalmente
b1 = np.zeros(hiddenDim)
print("\n*** W1\n",W1)
print("\n*** b1\n",b1)

#Pesos input to hidden W2 y b2
W2 = np.random.randn(hiddenDim,outputDim) #floats distrubuidos normalmente
b2 = np.zeros(outputDim)
print("\n*** W2\n",W2)
print("\n*** b2\n",b2)


#**** Feedforward
# De la entrada a la capa oculta
z1 = X.dot(W1) + b1#net
f1 = np.array(1/(1 + np.exp(-z1))) #sigmoide

#De la capa oculta a la capa de salida
z2 = f1.dot(W2) + b2 
#f2 = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True) #Activación Softmax sumar por renglones (axis = 1)
f2 = np.array(1/(1 + np.exp(-z2))) #sigmoide

print("*** Final probabilities ***")
print(f2) #Por cada patron de entrada hay una salida de su predicción



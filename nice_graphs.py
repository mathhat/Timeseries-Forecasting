from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
np.random.seed(1)
plt.style.use(['ggplot','Solarize_Light2','bmh'])

n = 50
k = 25
x = np.arange(n)
y = x/2+np.random.random(n)*7

plt.title("Temperature Dataset Split into Two")
plt.scatter(x[:k],y[:k],s=100,color="r",label="training set")#,alpha=0.5)#,label="training set")
plt.scatter(x[k:],y[k:],s=100,color='#0ee332',label="test set")#,alpha=0.5)# o',label="test set")
plt.ylabel("Average Temperature [Celsius]")
plt.xlabel("Time [Days]")
#plt.show()
x=x.reshape((-1,1))
model = LinearRegression()
model.fit(x[:k],y[:k])
#plt.plot(x[:k],model.predict(x[:k]),'r',label="model")
#plt.plot(x[k:],model.predict(x[k:]),'g',label="model's prediction")
plt.legend()

plt.show()

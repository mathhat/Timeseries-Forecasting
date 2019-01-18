from utilize import fetch , multi_pred
from model import train
import matplotlib.pyplot as plt

permutation = "0 0 0 0 0 0 0 0" #['TT', 'JY', 'TY', 'FT', 'JT', 'TZ', 'PD', 'PR']
place = 'VIK'
path_to_places = "/home/josephkn/Documents/Fortum/master/weatherdata/"
placepred = 1 #switch that activates place based model
timesteps = 10; future = 1; time_interval = 600
activation = 'relu'; epochs=4;multipred=1; nodes=128;batch_size=64;
weather = 0 #multipred must also be true for weatherdata to be used in model
time = 0
runs = 5
if multipred == 0 or placepred:
    permutation = 'none'
for i in range(runs):
    model, testx, testy,pdt_index = train( #trains model based on the variable above
        timesteps,future,time_interval,Permutation=permutation,place=place,activation=activation,
        preserve=0,
        patience=0,
        smooth=0,
        epochs=epochs,
        multipred=multipred,
        nodes1=nodes,
        nodes2=nodes,
        batch_size=batch_size,
        weather=weather,
        time=time,
        path=path_to_places)
'''
model,testx,testy,pdt_index = fetch(timesteps,future,time_interval,
                                                    Permutation=permutation,
                                                    smooth=0,
                                                    epochs=epochs,
                                                    multipred=multipred,
                                                    nodes1=nodes)
if testy.shape[0] > 3000:
    k=3000
    testx = testx[:k]
    testy = testy[:k]
ks = multi_pred(model,testx,timesteps,time_interval,pdt_index)
colors = ['cyan','magenta','red','black','green','blue']
for i in range(0,testy.shape[0]-timesteps-1,timesteps):
    plt.plot(range(i,i+timesteps),ks[i:(i+timesteps)],colors[np.random.randint(0,len(colors))])
#k = model.predict(testx.reshape((testx.shape[0],testx.shape[2],-1)))
plt.plot(testy[:,pdt_index],label='truth')
plt.legend()
plt.show()
'''

import tensorflow as tf
import matplotlib.pyplot as plt
from tools import *
from load_multi_pred import load_perm
from sys import exit
from collections import deque
from model import train
#timesteps = how many time increments to feed into the model
#future_vision = how many time increments into the future you wanna look
#time_interval = seconds, time increment size
#batch_size = how many predictions per iteration
#epochs = how many time to go through the data completely
#nodes1 = nodes in the input vector of the lstm
#nodes2 = nodes in the fully connected layer outside of the lstm
#validation_percentage how many percent of the data should be put in the validation set
#test_percentage = ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ test set
#patience = patience is how many times during training we allow the epoch no return a non improving MSE before cancelling the training

def fetch(timesteps,future_vision,time_interval,
                                    test_percentage=10,
                                    epochs = 1,
                                    Permutation='06211000',
                                    nodes1=128,
                                    patience=0,
                                    multipred=1,
                                    smooth=0):
    datax,datay,pdt_index,permutation = load_perm(time_interval,
                                                  future_vision,
                                                  timesteps,
                                                  Permutation,
                                                  smooth=smooth,
                                                  multipred=multipred)

    test_cursor = int((1-test_percentage/100)*datax.shape[0]) #where to split the data
    testy = datay[test_cursor+timesteps:]
    testx = datax[test_cursor+timesteps:]
    testx = testx.reshape((list(testx.shape[0:1])+[1]+list(testx.shape[1:])))

    if multipred==0: pdt_index=0


    model_name ='/home/josephkn/Documents/Fortum/master/models/'
    model_name += '%dtimesteps_%dtimeinterval_%dforward_%de_%dn_%s_multipred=%d'%(timesteps,time_interval,future_vision,epochs,nodes1,Permutation,multipred)
    try :
        f = open(model_name,'r')
        f.close()
        model = tf.keras.models.load_model(model_name)
        return model,testx,testy,pdt_index
    except:
        FileNotFoundError: print('model does not exist, go train it. exiting')
        exit()

def multi_pred (model,tests,timesteps,time_interval,pdt_index):
    future = timesteps
    in_out = deque(testx[0,0,:],timesteps)
    ks = []
    placehold = np.asarray(in_out)
    shape = list(placehold.shape)
    placehold = np.zeros(([1]+shape))
    for i in range(len(tests)-1):
        if i % future == 0:
            in_out = deque(testx[i+1,0,:],timesteps)
            placehold[0] = in_out
            k = model.predict(placehold)
            ks.append(k[0][pdt_index])
        else:
            in_out.append(k[0])
            placehold[0] = in_out
            k = model.predict(placehold)
            ks.append(k[0][pdt_index])
    return ks
permutation = "06211000"
timesteps = 50; future = 1; time_interval = 60;activation = 'linear'; epochs=3;multipred=0;
train(timesteps,future,time_interval,Permutation=permutation,activation=activation,preserve=1,patience=0,smooth=1,epochs=epochs,multipred=multipred)#,nodes1=256,nodes2=256)
model,testx,testy,pdt_index = fetch(timesteps,future,time_interval,Permutation=permutation,smooth=1,epochs=epochs,multipred=multipred)
if testy.shape[0] > 1000:
    k=1000
    testx = testx[:1000]
    testy = testy[:1000]
ks = multi_pred(model,testx,timesteps,time_interval,pdt_index)
colors = ['cyan','magenta','red','black','green','blue']
for i in range(0,testy.shape[0]-timesteps-1,timesteps):
    plt.plot(range(i,i+timesteps),ks[i:(i+timesteps)],colors[np.random.randint(0,len(colors))])
#k = model.predict(testx.reshape((testx.shape[0],testx.shape[2],-1)))
plt.plot(testy[:,pdt_index],label='truth')
plt.legend()
plt.show()

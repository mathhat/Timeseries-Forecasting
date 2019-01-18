import tensorflow as tf
import numpy as np
from tools import *
from load_multi_pred import load_perm
from sys import exit
from subprocess import call
import matplotlib.pyplot as plt
'''
timesteps = how many time increments to feed into the model
#future_vision = how many time increments into the future you wanna look
#time_interval = seconds, time increment size
#batch_size = how many predictions per iteration
#epochs = how many time to go through the data completely
#nodes1 = nodes in the input vector of the lstm
#nodes2 = nodes in the fully connected layer outside of the lstm
#validation_percentage how many percent of the data should be put in the validation set
#test_percentage = ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ test set
#patience = patience is how many times during training we allow the epoch no return a non improving MSE before cancelling the training
'''
def train(timesteps,future_vision,time_interval,batch_size=64,
                                                validation_percentage=2,
                                                test_percentage=10,
                                                activation = 'relu',
                                                epochs = 1,
                                                Permutation='none',#'0 6 2 1 1 0 0 0',
                                                place="none",
                                                savemod=1,
                                                save_img=0,
                                                save_bench=1,
                                                nodes1=128,
                                                nodes2=128,
                                                patience=0,
                                                multipred=1,
                                                preserve = 1,
                                                weather = 0,
                                                smooth=0,
                                                time=0,
                                                path = '/'):
    try:
        f =open('/home/josephkn/Documents/Fortum/master/models/%dtimesteps_%dtimeinterval_%dforward_%de_%dn_%s_multipred=%d'%(timesteps,time_interval,future_vision,epochs,nodes1,Permutation,multipred))
        f.close()
        print('you have this model')
        if preserve==1:
            return 0
        else:
            injpiu
    except:
        print('Initializing training of permutation %s'%Permutation)

    datax,datay,pdt_index,pdt_tag,means0,means,stds = load_perm(time_interval,future_vision,timesteps,Permutation,place,smooth,multipred,weather,time,path)

    test_cursor = int((1-test_percentage/100)*datax.shape[0]) #where to split the data
    testx = datax[test_cursor+timesteps:]
    testy = datay[test_cursor+timesteps:]
    datax = datax[:test_cursor+1]
    datay = datay[:test_cursor+1]
    datay = datay[:,pdt_index][:,np.newaxis]
    testy = testy[:,pdt_index][:,np.newaxis]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(nodes1,kernel_initializer='glorot_normal'
                                        ,input_shape=datax.shape[1:],
                                        return_sequences=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dense(nodes2,kernel_initializer='glorot_normal',
                                    activation=activation,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    if multipred:
        model.add(tf.keras.layers.Dense(datay.shape[-1],
                                        kernel_initializer='glorot_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        activation='linear'))
    else:
        model.add(tf.keras.layers.Dense(1,kernel_initializer='glorot_normal',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        activation='linear'))

    model.compile(loss='mean_squared_error', optimizer= 'adam')
    if multipred:
        history = model.fit(datax, datay, validation_split=validation_percentage/100,
                            epochs=epochs, batch_size=batch_size, verbose=1,shuffle=0,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience)])#,validation_data=(X_valid, y_valid))
        scores = model.evaluate(testx, testy, verbose=0)
    else:
        history = model.fit(datax, datay[:,pdt_index], validation_split=validation_percentage/100,
                            epochs=epochs, batch_size=batch_size, verbose=1,shuffle=0,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience)])
        scores = model.evaluate(testx, testy[:,pdt_index], verbose=0)

    print('average mse:', scores)
    if savemod:
        path = '/home/josephkn/Documents/Fortum/master/models/'
        model.save(path+'%dtimesteps_%dtimeinterval_%dforward_%de_%dn_%s_multipred=%d'%(timesteps,
                                                                                        time_interval,
                                                                                        future_vision,
                                                                                        epochs,
                                                                                        nodes1,
                                                                                        Permutation,
                                                                                        multipred))

    if save_bench and multipred:
        with open('bench/benchmarks_%dinterval_multipred.txt'%(time_interval),'a')as f:
            f.write('%f %d %d %d %s %d %d %d\n'% (scores,patience,nodes1,timesteps,
                                            Permutation,future_vision,weather,time))

    if save_bench and multipred==0:
        k = model.predict(testx).squeeze()
        gay =np.cumsum(testy*stds[0]+means[0])+means0[0]
        plt.plot(gay)
        plt.plot(np.cumsum(testx[:,-1,0]*stds[0]+means[0]) + k*stds[0] +means0[0],label="model")


        print(np.mean(abs(k-testy)), 'model abs error')
        print(np.mean(abs(testy[1:]-testy[0:-1])), 'copy last step abs error')

        plt.legend()
        plt.show()
        exit()
        errors = (k-testy)**2
        error_sort = np.sort(errors)
        median = np.median(error_sort)
        with open('bench/benchmarks_%dinterval_unipred.txt'%(time_interval),'a')as f:
            f.write('%f %f %d %d %d %s %d\n'% (scores,median,patience,nodes1,timesteps,Permutation,future_vision))

    if save_img:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('imgs/%s.jpg'%Permutation)
    del datax, datay,history,test_cursor
    return model, testx, testy,pdt_index

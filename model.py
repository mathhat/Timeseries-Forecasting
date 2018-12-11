import tensorflow as tf
import numpy as np
from tools import *
from load_multi_pred import load_perm
from sys import exit
from subprocess import call
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

def train(timesteps,future_vision,time_interval,batch_size=64,
                                                validation_percentage=2,
                                                test_percentage=10,
                                                activation = 'relu',
                                                epochs = 1,
                                                Permutation='06211000',
                                                savemod=1,
                                                save_img=0,
                                                save_bench=1,
                                                nodes1=128,
                                                nodes2=128,
                                                patience=0,
                                                multipred=1,
                                                preserve = 1,
                                                smooth=0):
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

    datax,datay,pdt_index,permutation = load_perm(time_interval,future_vision,timesteps,Permutation,smooth,multipred)

    test_cursor = int((1-test_percentage/100)*datax.shape[0]) #where to split the data
    testx = datax[test_cursor+timesteps:]
    testy = datay[test_cursor+timesteps:]
    datax = datax[:test_cursor+1]
    datay = datay[:test_cursor+1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(nodes1,kernel_initializer='normal',input_shape=datax.shape[1:],return_sequences=True))
    model.add(tf.keras.layers.CuDNNLSTM(nodes1,kernel_initializer='normal',input_shape=datax.shape[1:],return_sequences=True))
    model.add(tf.keras.layers.CuDNNLSTM(nodes1,kernel_initializer='normal',input_shape=datax.shape[1:],return_sequences=False))
    #model.add(tf.keras.layers.CuDNNGRU(nodes1,kernel_initializer='uniform',input_shape=datax.shape[1:]))
    #keras.layers.ConvLSTM2D(32, timesteps,
    model.add(tf.keras.layers.Dense(nodes2,kernel_initializer='normal',activation=activation))
    if multipred:
        model.add(tf.keras.layers.Dense(datax.shape[-1],kernel_initializer='normal'))
    else:
        model.add(tf.keras.layers.Dense(1,kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer= 'adam')
    if multipred:
        history = model.fit(datax, datay, validation_split=validation_percentage/100, epochs=epochs, batch_size=batch_size, verbose=1,shuffle=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience)])#,validation_data=(X_valid, y_valid))
        scores = model.evaluate(testx, testy, verbose=0)
    else:
        history = model.fit(datax, datay[:,pdt_index], validation_split=validation_percentage/100, epochs=epochs, batch_size=batch_size, verbose=1,shuffle=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=patience)])#,validation_data=(X_valid, y_valid))
        scores = model.evaluate(testx, testy[:,pdt_index], verbose=0)

    print('average mse:', scores)
    if savemod:
        model.save('/home/josephkn/Documents/Fortum/master/models/%dtimesteps_%dtimeinterval_%dforward_%de_%dn_%s_multipred=%d'%(timesteps,time_interval,future_vision,epochs,nodes1,Permutation, multipred))

    if save_bench and multipred:
        with open('bench/benchmarks_%dinterval_multipred.txt'%(time_interval),'a')as f:
            f.write('%f %d %d %d %s %d\n'% (scores,patience,nodes1,timesteps,Permutation,future_vision))

    if save_bench and multipred==0:
        k = model.predict(testx)
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
        plt.savefig('%s.jpg'%permutation)
    del datax, datay, testx,testy,history,test_cursor
    return 0

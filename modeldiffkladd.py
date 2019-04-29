import tensorflow as tf
import numpy as np
from tools import *
from load_multi_pred import load_perm, load_features,load_special_features
from sys import exit
from subprocess import call
from dateframe import holidays
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Input, CuDNNLSTM,Dense,Dropout,Embedding, Flatten
from tensorflow.keras.models import Model, Sequential
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
#weather, if you want weater data, it's added anyways I think
#smooth, this doesn't actually work
#time, time
#differentiate, makes most of the data into its time derivative, but not really
'''
def train(timesteps,future_vision,time_interval,batch_size=64,
                                                validation_percentage=5,
                                                test_percentage=20,
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
                                                differentiate =0,
                                                method ='f_score',
                                                optimized_features=0,
                                                k=20,
                                                categorical=0,
                                                catkeys=["weekday"],
                                                holiday=0,k2=0,special=0):
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
    if optimized_features and multipred:
        arrays,tags,pdt_index,pdt_tag,permutation,start,end = load_features(time_interval,place,method,k)
        if special:
            specialarrays,specialtags= load_special_features(time_interval,place,method,k2)
            for tag in specialtags:
                if tag not in tags:
                    arrays[tag]=specialarrays[tag]
            del specialarrays
    else:
        arrays,tags,pdt_index,pdt_tag,permutation,start,end = load_perm(time_interval,Permutation,place,smooth,multipred,weather)
    ##FIND OUT WHERE TO DIFFERENTIATE AND SCALE
    pdt_original = arrays[pdt_tag].copy()
    if differentiate:
        Arrays = make_differential_data2(arrays,future_vision,tags)
    else:
        Arrays = np.zeros((len(tags),len(arrays[tags[0]])),dtype=np.float32)
        for i in range(len(tags)):
            Arrays[i] =  arrays[tags[i]].squeeze()
    del arrays
    #here we create non-categorical time data
    dims = list(range(Arrays.shape[0]))
    intercepts=[]
    means=[]
    stds=[]
    for i in dims:
        mean = Arrays[i].mean() #iurdngswvlgeufzboingxegbldzuidxjocfkpvålj
        std = Arrays[i].std()
        Arrays[i] -= mean
        Arrays[i] /= std
        means.append(mean)
        stds.append(std)
    if time:
        timedata = create_time_variables(start,end,time_interval)
        if categorical == 0:
            noncat = noncategorical_timedata(timedata)
            for key in noncat.keys():
                if differentiate:
                    Arrays = np.vstack((Arrays,noncat[key][:-(future_vision+1)]))
                else:
                    Arrays = np.vstack((Arrays,noncat[key]))
        else:
            cat = categorical_timedata(timedata,catkeys)
            if holiday:
                holidaysArr = holidays(time_interval)[:,None]
                cat = np.concatenate((cat,holidaysArr),axis=1)
            if differentiate:
                cat = cat[:-1]
            print("categorical shape: ",cat.shape)

    dims = list(range(Arrays.shape[0]))
    if categorical==0 and holiday == 1:
        cat = holidays(time_interval)[:,None]
    indices = extract_indices2(timesteps,future_vision,[],Arrays)
    indices = np.asarray(indices)
    ranger = np.arange(len(indices))
    datax = np.zeros((len(indices),timesteps,Arrays.shape[0]),dtype=np.float)
    datay = np.zeros((len(indices),Arrays.shape[0]),dtype=np.float32)


    #check = Arrays[0].copy()
    #print(check.shape)
    X = [extract_samples2(x,timesteps,indices,datax,datay,ranger,Arrays[x]) for x in dims]

    del X,Arrays,ranger,indices
    plt.plot(datax[:,-1,pdt_index+len(tags)])
    plt.show()
    exit()
    test_cursor = int((1-test_percentage/100)*datax.shape[0]) #where to split the data
    print(test_cursor)
    testx = datax[test_cursor:].copy()#+timesteps:]
    testy = datay[test_cursor:].copy()#+timesteps:]
    datax = datax[:test_cursor]
    datay = datay[:test_cursor]
    datay = datay[:,pdt_index][:,np.newaxis]
    testy = testy[:,pdt_index][:,np.newaxis]
    if holiday or categorical:
        cat_test = cat[(future_vision+timesteps+test_cursor-1):]
        catx = cat[:test_cursor]
    input1 = Input(shape=(datax.shape[1:]))
    lstm = CuDNNLSTM(nodes1,kernel_initializer='glorot_normal'
                                        ,input_shape=datax.shape[1:],
                                        return_sequences=False)(input1)
    dropout = Dropout(0.25)(lstm)
    if categorical or holiday:
        input2 = Input(shape=(cat.shape[-1],))
        embedding = Embedding(batch_size,64)(input2)

        embeddingflat = Flatten()(embedding)

        embed_dense = Dense(32,kernel_initializer='glorot_normal',
                            activation=activation)(embeddingflat)
        marge = Concatenate()([lstm,embed_dense])
        dropout = Dropout(0.25)(marge)
        outputs = Dense(1,kernel_initializer='glorot_normal',
                            activation='linear')(dropout)
        model = Model(inputs=[input1,input2],outputs=outputs)

    else:
        outputs = Dense(1,kernel_initializer='glorot_normal',
                            activation='linear')(dropout)


        model = Model(inputs=input1,outputs=outputs)
    '''
    model = Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(nodes1,kernel_initializer='glorot_normal'
                                        ,input_shape=datax.shape[1:],
                                        return_sequences=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.00)))
    #model.add(tf.keras.layers.Dense(nodes2,kernel_initializer='glorot_normal', #want dense before or after merge?
    #                                activation=activation,
    #                                kernel_regularizer=tf.keras.regularizers.l2(0.00)))

    model.add(tf.keras.layers.Dropout(0.25))
    if categorical==0 and holiday ==0:
        model.add(tf.keras.layers.Dense(datay.shape[-1],
                                    kernel_initializer='glorot_normal',
                                    activation='linear'))
    '''
    filepath = '/home/josephkn/Documents/Fortum/master/models/best.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='min')

    model.compile(loss='mean_squared_error', optimizer= 'adam',metrics=['mse','mae'])
    if holiday or categorical:
        datax = [datax,catx]
        testx = [testx,cat_test]
        del cat

    history = model.fit(datax, datay, validation_split=validation_percentage/100,
                        epochs=epochs, batch_size=batch_size, verbose=1,shuffle=0,
                        callbacks=[checkpoint])#tf.keras.callbacks.EarlyStopping(patience=patience)])
    kk = model.predict(testx).squeeze() #predicted testy

    model.load_weights(filepath)
    model.compile(loss='mean_squared_error', optimizer= 'adam',metrics=['mse','mae'])
    scores = model.evaluate(x=testx,y=testy, verbose=0)
    print('average mse:', scores)
    '''
    if savemod:
        path = '/home/josephkn/Documents/Fortum/master/models/'
        model.save(path+'%dtimesteps_%dtimeinterval_%dforward_%de_%dn_%s_multipred=%d'%(timesteps,
                                                                                        time_interval,
                                                                                        future_vision,
                                                                                        epochs,
                                                                                        nodes1,
                                                                                        Permutation,
                                                                                        multipred))
    '''
    if save_bench and multipred and place=='none':
        with open('bench/benchmarks_%dinterval_multipred.txt'%(time_interval),'a')as f:
            f.write('%f %d %d %d %s %d %d %d\n'% (scores,patience,nodes1,timesteps,
                                            Permutation,future_vision,weather,time))

    if save_bench and multipred and place!='none':###########ERRORS ARE BEHAVING NORMALLY!
        method_=''
        if optimized_features:
            dir = "bench_multi_lstm"
            additional='_k=%d'%k
            method_ = method
        else:
            dir = "bench_place_sparse"
            additional =''
            permutation = ''.join((str(i)+',') for i in permutation.tolist())
        print(model.metrics_names)
        mse,mae = scores[1:]

        errors = (kk-testy.squeeze())
        abserrors = abs(errors)
        error_sort = np.sort(abserrors)
        median = np.median(error_sort)*stds[pdt_index]
        physical_error = abserrors.mean()*stds[pdt_index]
        '''
        if differentiate:
            dataX, dataY, copy_last, manual_physical= differencial_error_check(
                testy.shape[0],timesteps,
                future_vision,pdt_original,
                pdt_index,kk,physical_error,
                stds,means,dims)
        '''
        #else:
        dataX, dataY, copy_last, manual_physical = error_check(
            testy.shape[0],timesteps,
            future_vision,pdt_original,
            pdt_index,kk,physical_error,
            stds,means,dims)
        print(stds[pdt_index],'std')
        print(abs(errors).mean(),'mae')
        print(median,'median abs')
        print(physical_error,'physical')
        print(dir+'/benchmarks_%s_%dinterval_%dsteps_multipred%s.txt'%(place,time_interval,timesteps,additional))
        with open(dir+'/benchmarks_%s_%dinterval_%dsteps_multipred%s.txt'%(place,time_interval,timesteps,additional),'a')as f:
            f.write('%f %f %f %f %d %d %d %s %d %d %d %s %d %d\n'% (physical_error,mse,median,copy_last,patience,nodes1,timesteps,
                                            permutation[:-1],future_vision,weather,time,method_,differentiate,k2))
    if save_bench and multipred==0:
        #print(np.mean(abs(k-testy)), 'model abs error')
        errors = (testy.squeeze()-kk)
        error_sort = np.sort(abs(errors))
        mse,mae = scores[1:]
        physical_errors = (np.abs(errors)*stds[pdt_index]) #shortcut
        physical_error = physical_errors.mean()
        median = np.median(np.sort(physical_errors))
        print(physical_error)
        if differentiate:
            dataX, dataY, copy_last, manual_physical= differencial_error_check(
                testy.shape[0],timesteps,
                future_vision,pdt_original,
                pdt_index,kk,physical_error,
                stds,means,dims)
        else:
            dataX, dataY, copy_last, manual_physical = error_check(
                testy.shape[0],timesteps,
                future_vision,pdt_original,
                pdt_index,kk,physical_error,
                stds,means,dims)
        '''
        plt.plot(dataX,label="x")
        plt.plot(dataY,label="y")
        plt.show()
        '''
        print(physical_error, " physical error (shortcut)")
        print(manual_physical, " physical error (manual)")
        print(copy_last, " copy last step error")
        dir = "alternative_bench/lstm"
        if activation == "sigmoid":
            dir+="2"
        with open(dir+'/benchmarks_%dinterval_unipred.txt'%(time_interval),'a')as f:
            f.write('%f %f %f %f %d %d %d %s %d %d\n'% (physical_error,mse,median,copy_last,patience,nodes1,timesteps,Permutation,future_vision,differentiate))

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

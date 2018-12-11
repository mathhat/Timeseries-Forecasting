from tools import *
import pandas as pd
import matplotlib.pyplot as plt

def load_perm(time_interval,future,kernel_size,permutation,smooth=0,multipred=0):
    labeltag = "VIK_PDT2002.vY"
    path = "/home/josephkn/Documents/Fortum/master/pickle5/"
    arrays = dict()
    tags=[]
    units = []

    #extract tag names and unit types
    with open('tags_updated.txt','r') as f:
        for line in f:
            tags.append(line[:-1].split(' ')[0])
            if line[4:6] not in units:
                units.append(line[4:6])

    permutations = permutation
    permutations = [int(i) for i in permutations]
    if multipred:
        n_units = len(units)
        arrays = load_permutation(n_units,units,tags,arrays,path,time_interval,permutations)
        tags = list(arrays.keys())
        arrays = cross_reference_datetimes(tags,arrays)
    else:
        n_units = 1
        for tag in tags:
            if "PDT" in tag:
                arrays = dict()
                arrays[tag] =  pickle_load(path+tag+'.pickle')
                arrays[tag]=arrays[tag].resample('%ds'%time_interval).mean()
                arrays[tag]=arrays[tag].fillna(method='ffill')
                tags=[tag]
                break

    FROM = pd.to_datetime('090117') #format is mo da yr ######## here's the date hack
    TO = pd.to_datetime('040118')
    ind = arrays[tags[0]].index
    for i in range(len(ind)):
        if ind[i] == FROM:
            start=i
            break
    for i in range(len(ind)-1,0,-1):
        if ind[i] == TO:
            end=i
            break
    if smooth:
        for tag in tags:
            win = 1200//time_interval
            print('win: ',win)
            plt.plot(arrays[tag]+1,alpha=0.8,label='truth',)
            arrays[tag]=arrays[tag].rolling(window=win,win_type='gaussian',center=True).mean(std=1)
            plt.plot(arrays[tag],label='smooth')
            plt.legend()
            plt.show()
            print(arrays[tag].head(),arrays[tag].tail())
            exit()

    arrays = remove_time_aspect(arrays,start,end)
    Arrays = np.zeros((n_units,len(arrays[tags[0]])),dtype=np.float32)
    i=0
    for t in range(len(tags)):
        tag = tags[t]
        if 'PDT' in tag:
            pdt_index = i
        Arrays[i] = arrays[tag].squeeze()
        i+=1
    del arrays
    dims = list(range(Arrays.shape[0]))

    '''
    if smooth:
        win = 1200//time_interval
        #arrays = np.zeros(([len(dims)]+[Arrays.shape[1]-(win-1)]))
        arrays = np.zeros(([len(dims)]+[Arrays.shape[1]]))
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0))
            return (cumsum[N:] - cumsum[:-N]) / float(N)
        for i in dims:
            arrays[i] = running_mean(Arrays[i],win)
        Arrays = arrays
        del arrays
    #import parmap
    '''
    indices = extract_indices2(kernel_size,future,[],Arrays) #fixes arrays to
                                                             #same timefra
    indices = np.asarray(indices)
    ranger = np.arange(len(indices))
    dataX = np.zeros((len(indices),kernel_size,n_units),dtype=np.float)
    datay = np.zeros((len(indices),n_units),dtype=np.float32)
    X = [extract_samples2(x,kernel_size,indices,dataX,datay,ranger,Arrays[x]) for x in dims]
    #X = parmap.map(extract_samples2,dims,kernel_size,indices,dataX,datay,ranger,Arrays)
    del X,Arrays,indices,ind

    dataX_mean = np.mean(dataX,axis=(1,0)).squeeze()

    datay_mean = np.mean(datay,axis=(1,0)).squeeze()

    dataX_std = np.std(dataX,axis=(1,0)).squeeze()
    datay_std = np.std(datay,axis=(1,0)).squeeze()
    dataX = (dataX-dataX_mean)/dataX_std
    datay = (datay-datay_mean)/datay_std
    return dataX, datay,pdt_index,permutations

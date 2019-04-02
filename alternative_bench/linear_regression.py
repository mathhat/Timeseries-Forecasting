# Importing various packages
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from load_multi_pred import load_perm
from sklearn.linear_model import LinearRegression
from tools import *
def parse(dirpath,timesteps,interval,differentiated,future_vision,copylast=0,median=0):
    m = []
    c = []
    med = []
    with open('%s/benchmarks_%dinterval_unipred.txt'%(dirpath,interval),'r') as f:
        args = f.readline().split(',')
        try:
            diff = args.index('differentiate')
        except:
            diff= args.index('differentiate\n')
        steps = args.index('timesteps')
        future = args.index('future_vision')
        for line in f:
            line = line.split(' ')
            if int(line[diff][0]) == differentiated and int(line[steps]) == timesteps and int(line[future]) == future_vision:
                m.append(float(line[0]))
                if copylast:
                    c.append(float(line[3]))
                if median:
                    med.append(float(line[2]))
    if copylast and median:
        if median:
            return np.mean(m), np.mean(c),np.mean(med)
    elif copylast:
        return np.mean(m), np.mean(c)
    elif median:
        return np.mean(m), np.mean(med)
    return np.mean(m)


def linreg_uni(time_interval,future_vision,timesteps,differentiate):
    Permutation ="none"
    place = "VIK"
    arrays,tags,pdt_index,pdt_tag,permutation,start,end = load_perm(time_interval,timesteps,Permutation,place)
    arrays[pdt_tag]
    pdt_original = arrays[pdt_tag].copy()

    tags = list(arrays.keys())
    if differentiate:
        Arrays = np.zeros((len(tags),len(arrays[tags[0]])-1),dtype=np.float32)
        Arrays = make_differential_data(arrays,future_vision,tags)
    else:
        Arrays = np.zeros((len(tags),len(arrays[tags[0]])),dtype=np.float32)
        for i in range(len(tags)):
            Arrays[i] =  arrays[tags[i]].squeeze()
    del arrays

    dims = list(range(Arrays.shape[0]))
    intercepts=[]
    means=[]
    stds=[]
    for i in dims:
        mean = Arrays[i].mean()
        std = Arrays[i].std()
        Arrays[i] -= mean
        Arrays[i] /= std
        means.append(mean)
        stds.append(std)
    indices = extract_indices2(timesteps,future_vision,[],Arrays)
    indices = np.asarray(indices)
    ranger = np.arange(len(indices))
    datax = np.zeros((len(indices),timesteps,Arrays.shape[0]),dtype=np.float)
    datay = np.zeros((len(indices),Arrays.shape[0]),dtype=np.float32)
    #check = Arrays[0].copy()
    #print(check.shape)
    X = [extract_samples2(x,timesteps,indices,datax,datay,ranger,Arrays[x]) for x in dims]

    del X,Arrays,ranger,indices

    test_cursor = int((1-test_percentage/100)*datax.shape[0]) #where to split the data
    testx = datax[test_cursor:].squeeze()#+timesteps:]
    testy = datay[test_cursor:]#+timesteps:]
    datax = datax[:test_cursor].squeeze()
    datay = datay[:test_cursor]
    datay = datay[:,pdt_index][:,np.newaxis]
    testy = testy[:,pdt_index][:,np.newaxis]


    linreg = LinearRegression()
    linreg.fit(datax,datay)


    #xnew = np.array([testx,testy])
    ypredict = linreg.predict(testx).squeeze()
    err = np.abs(ypredict-testy.squeeze())
    median = np.median(stds[pdt_index]*err)
    physerr = stds[pdt_index]*np.mean(err)
    pdt_original=pdt_original.squeeze()
    print(physerr)
    if differentiate:
        dataX, dataY, copy_last, manual_physical= differencial_error_check(
            testy.shape[0],timesteps,
            future_vision,pdt_original,
            pdt_index,ypredict,physerr,
            stds,means)
    else:
        dataX, dataY, copy_last, manual_physical = error_check(
            testy.shape[0],timesteps,
            future_vision,pdt_original,
            pdt_index,ypredict,physerr,
            stds,means)

    dir = "linreg_bench"
    with open(dir+'/benchmarks_%dinterval_unipred.txt'%(time_interval),'a')as f:
        f.write('%f %f %f %f %d %d %d\n'% (physerr,np.mean(err),median,copy_last,timesteps,future_vision,differentiate))

#plt.style.use(['ggplot','Solarize_Light2','bmh'])
'''#plot code
for diff in [0,1]:
    for time_interval in [60,120]:
        linreg = []
        time = []
        for future_vision in [1,2,3,4,5,10,15,20,25,30,35,40]:
            if time_interval*future_vision/60 > 40:
                continue
            time.append(time_interval*future_vision/60)
            for triplet in error_table[time_interval*future_vision]:
                if triplet[1] == time_interval and diff==triplet[2]:
                    linreg.append(triplet[0])
        plt.plot(time,linreg,label="diff = %d, dt = %d"%(diff,time_interval))
plt.plot(t,k,label="copy_last")
plt.title("Physical Error og (Univariate) Linear Regression Models")
plt.xlabel("Forecast Length [Minutes]")
plt.ylabel("Error [kPa]")
plt.legend()
plt.show()
'''
def linear_results():
    test_percentage = 20
    timesteps = 30
    error_table = dict()
    dirpath="linreg_bench"
    cps = dict()
    for differentiate in [0,1]:
        for time_interval in [60,120,300]:
            for future_vision in [1,2,3,4,5,10,15,20,25,30,35,40]:
                if time_interval*future_vision/60 > 40:
                    continue
                linreg, cp = parse(dirpath,timesteps,time_interval,differentiate,future_vision,copylast=1,median=0)
                try:
                    error_table[time_interval*future_vision].append([linreg,time_interval,differentiate])
                except:
                    error_table[time_interval*future_vision] = [[linreg,time_interval,differentiate]]
                cps[time_interval*future_vision] = cp
            if time_interval == 60:
                k=[];t=[]
                for i in sorted(list(cps.keys())):
                    if i/60 > 40:
                        continue
                    k.append(cps[i])
                    t.append(i/60)
    for diff in [0,1]:
        for time_interval in [60,120,300]:
            linreg = []
            time = []
            for future_vision in [1,2,3,4,5,10,15,20,25,30,35,40]:
                if time_interval*future_vision/60 > 40:
                    continue
                time.append(time_interval*future_vision/60)
                for triplet in error_table[time_interval*future_vision]:
                    if triplet[1] == time_interval and diff==triplet[2]:
                        linreg.append(triplet[0])
            if time_interval==120 and diff==0:
                return linreg,time,k,t
#linreg_uni(time_interval,future_vision,timesteps,differentiate)

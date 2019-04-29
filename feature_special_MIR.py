import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sub
from tools import pickle_load, tags_of_place, remove_time_aspect,load_weather
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression,mutual_info_regression

#import matplotlib.pyplot as plt
place = "VIK"
path = "/home/josephkn/Documents/Fortum/master/%s_pickles2/"%place
#df = pickle_load(path2+place+'6.pickle')
#grp = df.groupby('tag',sort=False, as_index=False)
arrays = dict()
tags=[]
#del df
k=0
time_interval = 120
try:
    with open(path+"tags_winter.txt",'r') as f:
        count = 0
        j=0
        for line in f:
            line= line[:-1]
            tags.append(line)
            count+=1
            if count==25: #set number to 25 if you only wanna see correlation of special tags with pdt2002
                j=100
                line='VIK_PDT2002.vY'
            df = pickle_load(path+line+".pickle")

            try:
                df=df.resample('%ds'%time_interval).fillna(method='ffill')
            except:
                j+=1
                df=df.resample('%ds'%time_interval).mean().fillna(method='ffill').fillna(method='bfill')
            arrays[line]=df
            if j==100:
                break
        print(arrays.keys())
except:
    print('k')
    with open(path+"tags.txt",'r') as f:
        j = 0
        tags= []
        for line in f:
            line=line[:-1]
            tags.append(line)
            df = pickle_load(path+line+".pickle")
            try:
                df=df.resample('%ds'%time_interval).fillna(method='ffill')
            except:
                j+=1
                df=df.resample('%ds'%time_interval).mean().fillna(method='ffill').fillna(method='bfill')
            arrays[line]=df
    k=1

#here we define what time of year we're interested in
FROM = pd.to_datetime('120117') #format is mo da yr ######## here's the date hack
TO = pd.to_datetime('020118')
labeltag = "VIK_PDT2002.vY"
tags = list(arrays.keys())
if k: #if we haven't saved the tags of this area, that are active during winter too
    for tag in tags[::-1]:
        ind = arrays[tag].index
        init = (FROM-ind[0]).total_seconds()
        end = (ind[-1]-TO).total_seconds()
        if init < 0 or end < 0:
            del arrays[tag]
            tags.remove(tag)
    with open(path+'tags_winter.txt','w') as f:
        for tag in tags:
            f.write(tag+'\n')


ind = arrays[labeltag].index
for i in range(len(ind)):
    if ind[i] == FROM:
        start=ind[i]
        break
for i in range(len(ind)-1,0,-1):
    if ind[i] == TO:
        end=ind[i]
        break

arrays = remove_time_aspect(arrays,start,end)
labels = arrays[labeltag]
labels = ((labels[1:].copy().squeeze()-labels.mean()) /labels.std()).astype(np.float32)
n_samples = len(labels)
n_features = len(tags)
Arrays = np.zeros((n_samples,n_features),dtype=np.float32)
i=0
for tag in tags:
    Arrays[:,i] = (arrays[tag][:-1].squeeze() - arrays[tag].mean())/arrays[tag].std()
    i+=1
del arrays
print(Arrays.shape,labels.shape)
print(Arrays.dtype,labels.dtype)
k = len(tags)
test = SelectKBest(score_func=mutual_info_regression,k=k)
fit = test.fit(Arrays, labels)
scores = fit.scores_
#from scipy.stats import rankdata
#ranks = rankdata(-scores, method='dense').astype(np.int32)
print(scores)
sort = np.argsort(scores).tolist()[::-1]
print(sort)
print(scores[sort[0]])
print(tags[sort[0]])

with open('k_best_special_features_weather_MIR_%d.txt'%time_interval,'w') as f:
    for i in range(k):
        f.write(str(scores[sort[i]])+', '+tags[sort[i]]+'\n')
#print(scores[sort[-1]])

#print(scores[np.argmin(ranks)])
#print(scores[np.argmax(ranks)])
print(min(scores))
print(max(scores))


with open(path+"units.txt",'a') as f:
    units = []
    for tag in tags:
        unit = str(tag[4:7])
        if unit[-1].isdigit():
            unit = unit[:-1]
        if unit not in units:
            units.append(unit)
            print(unit)
    for unit in sorted(units):
        f.write('%s\n'%unit)

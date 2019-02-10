#this script takes a pickled dataframe and pickles each dataframe within the dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sub
from tools import pickle_load, tags_of_place, remove_time_aspect,load_weather
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression,mutual_info_regression
#import matplotlib.pyplot as plt
time_interval = 300
place = "VIK"
sub.call("mkdir %s_pickles"%place,shell=True)
path = "/home/josephkn/Documents/Fortum/master/%s_pickles/"%place
path2 = "/home/josephkn/Documents/Fortum/master/pickle6/"
df = pickle_load(path2+place+'6.pickle')
grp = df.groupby('tag',sort=False, as_index=False)
arrays = dict()
tags=[]
del df
k=0
try:
    with open(path+"tags_winter.txt",'r') as f:
        for line in f:
            tags.append(line[:-1])

except:
    tags = list(tags_of_place(df))
    k=1


for tag,slicee in grp:
    slicee = slicee.drop(columns=['tag'])
    slicee['Date'] = pd.to_datetime(slicee['Date'])
    slicee = slicee.set_index('Date')
    slicee = slicee.sort_index()
    slicee=slicee.resample('%ds'%time_interval).fillna(method='ffill')
    arrays[tag] = slicee
weather_df = load_weather(time_interval)
keys =  list(weather_df.keys())
for key in keys:
    arrays[key] = weather_df[key]
    tags.append(key)
n_units = len(tags)
del weather_df
#here we define what time of year we're interested in
FROM = pd.to_datetime('090117') #format is mo da yr ######## here's the date hack
TO = pd.to_datetime('040118')
labeltag = "VIK_PDT2002.vY"


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
        start=i
        break
for i in range(len(ind)-1,0,-1):
    if ind[i] == TO:
        end=i
        break
arrays = remove_time_aspect(arrays,start,end)

labels = arrays[labeltag]
labels = (labels[1:].copy().squeeze()-labels.mean() /labels.std()).astype(np.float64)
n_samples = len(labels)
n_features = len(tags)
Arrays = np.zeros((n_samples,n_features),dtype=np.float64)
i=0
for tag in tags:
    Arrays[:,i] = (arrays[tag][:-1].squeeze() - arrays[tag].mean())/arrays[tag].std()
    i+=1
del arrays
print(Arrays.shape,labels.shape)
print(Arrays.dtype,labels.dtype)
k = 100
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
with open('k_best_features_weather_MIR_%d.txt'%time_interval,'w') as f:
    for i in range(k):
        f.write(str(scores[sort[i]])+', '+tags[sort[i]]+'\n')
#print(scores[sort[-1]])

#print(scores[np.argmin(ranks)])
#print(scores[np.argmax(ranks)])
print(min(scores))
print(max(scores))

'''
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
'''

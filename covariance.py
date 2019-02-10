#this script takes a pickled dataframe and pickles each dataframe within the dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sub
from tools import pickle_load, tags_of_place, remove_time_aspect,load_weather

#import matplotlib.pyplot as plt
time_interval = 600
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
    print('k')
    tags = list(tags_of_place(df))
    k=1
for tag,slicee in grp:
    slicee = slicee.drop(columns=['tag'])
    slicee['Date'] = pd.to_datetime(slicee['Date'])
    slicee = slicee.set_index('Date')
    slicee = slicee.sort_index()
    slicee=slicee.resample('%ds'%time_interval).fillna(method='ffill')
    if k==0:
        if tag in tags:
            arrays[tag] = slicee
        else:
            continue
    else:
        arrays[tag] = slicee

weather_df = load_weather(time_interval)
keys =  list(weather_df.keys())
for key in keys:
    arrays[key] = weather_df[key]
    tags.append(key)
n_units = len(tags)
print(len(tags))
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
labels = (labels[1:].copy().squeeze()-labels.mean() /labels.std())
n_samples = len(labels)
n_features = len(tags)
#Arrays = np.zeros((n_samples,n_features),dtype=np.float64)
tags = list(arrays.keys())
tags2=[]
for tag in tags:
    arrays[tag] = ((arrays[tag][:-1].squeeze() - arrays[tag][:-1].mean())/arrays[tag][:-1].std())
    tag = tag[4:]
    tag =tag.split('.')[0]
    tags2.append(tag)

tags.append("labels")
tags2.append("labels")
arrays["labels"]=labels
df = pd.DataFrame(arrays)
df.columns=tags2
del arrays

print(df.info())
corrmat = df.corr().abs().values
print(corrmat.shape)
args = np.argsort(corrmat[tags2.index("labels")])
with open('k_best_features_covar_%d.txt'%time_interval,'w') as f:
    for arg in args[::-1]:
        print(tags[arg])
        f.write(str(corrmat[tags2.index("labels"),arg])+', '+tags[arg]+'\n')

#exit()

import matplotlib.pyplot as plt
import seaborn as sns
#plt.figure(figsize=(20,20))
#plot heat map
ind=tags2.index("PDT2002")
print(ind)
#exit()
g=sns.heatmap(df.corr().abs(),cmap="RdYlGn",xticklabels=1,yticklabels=1)
plt.show()
'''
with open('k_best_features_weather_fscore_%d.txt'%time_interval,'w') as f:
    for i in range(k):
        f.write(str(scores[sort[i]])+', '+tags[sort[i]]+'\n')

'''

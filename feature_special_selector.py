import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess as sub
from tools import pickle_load, tags_of_place, remove_time_aspect,load_weather

#import matplotlib.pyplot as plt
time_interval = 60
place = "VIK"
path = "/home/josephkn/Documents/Fortum/master/%s_pickles2/"%place
#df = pickle_load(path2+place+'6.pickle')
#grp = df.groupby('tag',sort=False, as_index=False)
arrays = dict()
tags=[]
#del df
k=0

try:
    with open(path+"tags_winter.txt",'r') as f:
        count = 0
        j=0
        for line in f:
            line= line[:-1]
            tags.append(line[:-1])
            count+=1
            if count==25:
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
except:
    print('k')
    with open(path+"tags.txt",'r') as f:
        j = 0
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
        start=ind[i]
        break
for i in range(len(ind)-1,0,-1):
    if ind[i] == TO:
        end=ind[i]
        break
weather_df = load_weather(time_interval)
keys =  list(weather_df.keys())
for key in keys:
    arrays[key] = weather_df[key]
    tags.append(key)
n_units = len(tags)
print(len(tags))
del weather_df
arrays = remove_time_aspect(arrays,start,end)
arrays["wind_dir1"]=np.sin(arrays["wind_dir"])
arrays["wind_dir2"]=np.cos(arrays["wind_dir"])
labels = arrays[labeltag]
labels = (labels[1:].copy().squeeze()-labels.mean()) /labels.std()
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
with open('k_best_special_features_covar_%d.txt'%time_interval,'w') as f:
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
g=sns.heatmap(df.corr().abs(),cmap="RdYlGn")#,xticklabels=1,yticklabels=1)
plt.show()

with open('k_best_features_weather_fscore_%d.txt'%time_interval,'w') as f:
    for i in range(k):
        f.write(str(scores[sort[i]])+', '+tags[sort[i]]+'\n')

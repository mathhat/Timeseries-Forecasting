#this script takes a pickled dataframe and pickles each dataframe within the dataframe
import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sub
from tools import pickle_load, tags_of_place
#import matplotlib.pyplot as plt
place = "KLE"
sub.call("mkdir %s_pickles"%place,shell=True)
path = "/home/josephkn/Documents/Fortum/master/%s_pickles/"%place
path2 = "/home/josephkn/Documents/Fortum/master/pickle6/"
df = pickle_load(path2+place+'6.pickle')
grp = df.groupby('tag',sort=False, as_index=False)
tags = tags_of_place(df)

del df
with open(path+"tags.txt",'a') as f:
    for tag,slicee in grp:
        slicee = slicee.drop(columns=['tag'])
        slicee['Date'] = pd.to_datetime(slicee['Date'])
        slicee = slicee.set_index('Date')
        slicee = slicee.sort_index()
        slicee.to_pickle('%s.pickle'%tag)
    for tag in sorted(tags):
        f.write('%s\n'%tag)

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

from tools import *
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_perm(time_interval,future,kernel_size,permutation,place="none",smooth=0,multipred=0,weather=0,time=0):
    labeltag = "VIK_PDT2002.vY"
    path = "/home/josephkn/Documents/Fortum/master/pickle5/"
    path2 = "/home/josephkn/Documents/Fortum/master/pickle6/"
    path3 = "/home/josephkn/Documents/Fortum/master/"
    arrays = dict()
    tags=[]
    units = []

    #extract tag names and unit types
    with open('tags_updated.txt','r') as f:
        for line in f:
            tags.append(line[:-1].split(' ')[0])
            if line[4:6] not in units:
                units.append(line[4:6])

    if multipred and permutation != 'none' and place=='none':


        permutations = permutation
        permutations = [int(i) for i in permutations.split(' ')]
        n_units = len(units)
        arrays = load_permutation(n_units,units,tags,arrays,path,time_interval,permutations)
        tags = list(arrays.keys())
        arrays = cross_reference_datetimes(tags,arrays)
        n_units = len(tags)
    elif multipred and place!='none' and permutation =="none":
        tags = load_tags_of_place(path3,place)
        units = load_units_of_place(path3,place)
        n_units = len(units)
        permutation,tags_of_unit = create_data_permutation(n_units,units,tags)
        #here I randomly leave some of the permutation outside
        for i in range(len(permutation)):
            if np.random.randint(0,2):
                permutation[i] = -1

        #let's force pdt2002
        if place=='VIK':
            index_unit = units.index("PDT")
            index_tag = list(tags_of_unit["PDT"]).index(labeltag)
            permutation[index_unit]=index_tag

        arrays = load_permutation(len(units),units,tags,dict(),path3+"%s_pickles/"%place,time_interval, permutation)
        tags =list(arrays.keys())
        n_units = len(tags)
    elif multipred and place != 'none' and permutation != 'none':
        tags = load_tags_of_place(path3,place)
        units = load_units_of_place(path3,place)
        n_units = len(units)
        arrays = load_permutation(len(units),units,tags,dict(),path3+"%s_pickles/"%place,time_interval, permutation)
        tags =list(arrays.keys())
        n_units = len(tags)
    else:
        n_units = 1
        for tag in tags:
            if "PDT" in tag:
                arrays = dict()
                arrays[tag] =  pickle_load(path+tag+'.pickle')
                #arrays[tag]=arrays[tag].resample('s').mean()
                arrays[tag]=arrays[tag].resample('%ds'%time_interval).fillna(method='ffill')
                #arrays[tag]=arrays[tag].fillna(method='ffill')


                tags=[tag]
                break

    #here we import weather data
    if multipred and weather:
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
    for tag in tags:
        ind = arrays[tag].index
        init = (FROM-ind[0]).total_seconds()
        end = (ind[-1]-TO).total_seconds()
        if init < 0 or end < 0:
            del arrays[tag]
            tags.remove(tag)
            n_units-=1

    ind = arrays[tags[0]].index
    for i in range(len(ind)):
        if ind[i] == FROM:
            start=i
            break
    for i in range(len(ind)-1,0,-1):
        if ind[i] == TO:
            end=i
            break

    #here I remove data outside of the relevant timeframe
    arrays = remove_time_aspect(arrays,start,end)

    for t in range(len(tags)):
        tag = tags[t]
        if 'PDT' in tag:
            pdt_index = t #line below calculates the differenced series
    del ind

    return arrays,tags,pdt_index,tags[pdt_index],permutation
def load_features(time_interval,future,kernel_size,place,method='f_score',time=0,k=20):
    path = "/home/josephkn/Documents/Fortum/master/"
    time_interval_=time_interval
    if method=='MIR' and time_interval_ < 180:
        time_interval_ = 180
    methods = ["k_best_features_covar_%d.txt"%time_interval,
        "k_best_features_weather_MIR_%d.txt"%time_interval_,
        "k_best_features_weather_fscore_%d.txt"%time_interval]
    for files in methods:
        if method in files:
            break

    arrays = dict()
    tags=[]
    units = []
    with open(path+place+"_pickles/"+files,'r') as f:
        print("loading features chosen from "+files)
        line = f.readline() #initial line
        tag = line.split(' ')[-1]
        tags.append(tag[:-1])
        for line in f:
            line = line.split(' ')[-1]
            for tag in tags:
                if line.split('.')[0] == tag.split('.')[0]:
                    line='0'
                    break
            if line =='0':
                continue
            else:
                tag=line
                tags.append(tag[:-1])
            if (len(tags) == k and method!='covar') or (len(tags) == k+1):
                break
    weather_df = load_weather(time_interval)
    print(weather_df.columns)
    arrays = dict() #load best features
    for tag in tags[:]:
        if "." in tag:
            arrays[tag] =  pickle_load(path+place+"_pickles/"+tag+'.pickle')
            arrays[tag] =  arrays[tag].resample('%ds'%time_interval).fillna(method='ffill')
        elif tag == "labels":
            tags.remove(tag)
            continue
        else:
            arrays[tag] = weather_df[tag]
    del weather_df
    print(arrays.keys())

    #here we define what time of year we're interested in
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
    arrays = remove_time_aspect(arrays,start,end)
    return arrays,tags,0,tags[0],'none'

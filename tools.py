import pandas as pd
from collections import Counter
from datetime import datetime
from sys import exit
import numpy as np
import numba as nb
import multiprocessing
import os

'''
I asked for 10 min interval data from Blindern (other measuring stations are too inconsistent)
http://sharki.oslo.dnmi.no/portal/page?_pageid=73,39035,73_39057&_dad=portal&_schema=PORTAL
Here's an explanatory table over the weather data
     Code                                 Name     Unit
0      DD                  Wind direction (FF)  degrees
1      FF  Wind speed (10 meters above ground)      m/s
2  FG_010       Maximum gust (last 10 minutes)      m/s
3      RA                  Total precipitation       mm
4  RR_010                        Precipitation       mm
5  RT_010          Precipitation time (10 min)  minutes
'''

def load_weather(sek=300):
    '''loads weather data, interpolates it to the amount of seconds you'd like, then returns it in a df.'''
    pd_raw = pd.read_csv('file:///home/josephkn/Documents/Fortum/master/weatherdata/ten_min_good.csv')
    pd_raw = pd_raw.loc[:,pd_raw.columns!='St.no']
    pd_raw = pd_raw.loc[:,pd_raw.columns!='Unnamed: 0']
    new_columns = ['Date',
        'wind_dir',     #DD (see description at the top of the document)
        'wind',         #FF
        'max_gust',     #FG_010
        'tot_precip',   #RA
        'precip',       #RR_010
        'precip10']     #RT_010
    pd_raw.columns = new_columns
    pd_raw.loc[:,new_columns[1:]] = pd_raw[new_columns[1:]].interpolate()
    pd_raw[new_columns[0]]= pd.to_datetime(pd_raw[new_columns[0]],format="%d.%m.%Y-%H:%M")
    pd_raw = pd_raw.set_index('Date')
    df = pd_raw.resample('%ds'%sek).mean().interpolate()
    del pd_raw
    return df

def load_data(filename):
    '''loads fortum data into a dataframe with dates, tags and values'''
    pd_raw = pd.read_csv('file:///home/josephkn/Documents/Fortum/master/weatherdata/%s.zip'%filename,
        sep=',',
        na_filter=False,
        usecols=[1,2,3]
        ,float_precision=np.float32,
        low_memory=False,
        compression='zip')
    pd_raw.columns = ['Date','tag','Value'] #new names, then downcast from float64 to float32
    return pd_raw



def smart_read(path,nrows=30000,chunksize=50000,sensor_size=20000):
    '''loads zipped fortum district data into a dataframe with dates, tags and values'''
    chunks = []
    column_names = ['Date',
        'tag',
        'Value']

    dtype={'Date': str,
       'tag': str,
       'Value': np.float32}
    chunks = pd.read_csv('file://'+path,
        sep=',',
        na_filter=False,
        usecols=[1,2,3],
        names = column_names,
        dtype = dtype,
        chunksize=chunksize,
        header=None,
        low_memory=True,
        compression='zip')

    chunks = pd.concat(chunks)
    if nrows == 0:
        #print(chunks.info())
        return chunks
    tags = tags_of_place(chunks,sensor_size)
    chunks = chunks.loc[chunks['tag'].isin(tags)]
    #pd_raw.columns = ['Date','tag','Value']
    print(chunks.info())
    del tags
    return chunks


def smart_read_individual(filename):
    '''loads zipped fortum district data into a dataframe with dates, tags and values'''
    column_names = ['Date',
        'Value']
    dtype={'Date': str,
       'Value': np.float32}
    df = pd.read_csv('file:///home/josephkn/Documents/Fortum/master/taglist/%s'%filename,
        sep=',',
        na_filter=False,
        usecols=[1,3],
        names = column_names,
        dtype = dtype,
        header=None)
    return df

def pickle_load(path):
    return pd.read_pickle(path)

def tags_of_place(df,min_obs=0):
    ''''returns tags of the place you want, with a minimum amount of observations'''
    k = Counter(df['tag']) #dict object that with the # of observation of each tag
    if min_obs==0:
        return k.keys()
    tags = list()
    for tag in k:
        if ('vY' in tag[-3:])*(min_obs<k[tag]):
            tags.append(str(tag))
    del k
    return tags

def extract_arrays2(tags,df):
    '''return values in the dataframe which belongs to the tags'''
    arrays = dict()
    #df = df.sort_values(by='tag')
    grp = df.groupby('tag',sort=False, as_index=False)
    for tag,slicee in grp:
        slicee = slicee.drop(columns=['tag'])
        slicee['Date'] = pd.to_datetime(slicee['Date'])
        slicee = slicee.set_index('Date')
        slicee = slicee.sort_index()
        #slicee = slicee.resample('%ds'%time_interval).mean()
        arrays[tag] = slicee
    del grp,df
    return arrays

def smooth(df,win,gauss=0):
    if gauss:
        return df.rolling(window=win,win_type='gaussian').mean(std=1).shift(-win//2).iloc[win//2:-win//2]

    else:
        return df.rolling(window=win).mean().shift(-win//2).iloc[win//2:-win//2]

def most_freq(tag_type,tags,arrays,time_interval):
    relevant_tags = list()
    for tag in tags:
        if tag_type in tag[4:7]:
            relevant_tags.append(tag)
    if len(relevant_tags)==0:
        print("no relevant tags")
        return 0
    best_tag = relevant_tags[0]
    best_len = len(arrays[best_tag].resample('%ds'%time_interval).mean())
    if len(best_tag)>1:
        for i in range(1,len(relevant_tags)):
            tag = relevant_tags[i]
            length = len(arrays[tag])
            if length > best_len:
                best_len = length
                best_tag = tag
    return best_tag

def most_freq_plural(units,tags,arrays,time_interval):
    relevant_tags = []
    for unit in units:
        relevant_tags.append(most_freq(unit,tags,arrays,time_interval))
    for i in range(len(relevant_tags))[::-1]:
        if relevant_tags[i]==0:
            relevant_tags.pop(i)
    return relevant_tags

def cross_reference_datetimes(relevant_tags,arrays):
    earliest, latest = arrays[relevant_tags[0]].index[0],arrays[relevant_tags[0]].index[-1]
    for tag in relevant_tags[1:]:
        mask = arrays[tag].notna()
        if (mask.index[0]-earliest).total_seconds() > 0:
            earliest = mask.index[0]
        if (mask.index[-1]-latest).total_seconds() < 0:
            latest = mask.index[-1]

    for i in range(len(relevant_tags)):
        arrays[relevant_tags[i]]= arrays[relevant_tags[i]][earliest:latest]
    return arrays
def create_time_variables(start,end,time_interval):
    time = np.arange(end-start)*time_interval
    steps_in_a_day = (24*3600.)
    sin_day = np.sin(time*2*np.pi/steps_in_a_day)
    cos_day = np.cos(time*2*np.pi/steps_in_a_day)
    return sin_day,cos_day

def remove_time_aspect(arrays,start,end):
    tags =(list(arrays.keys()))
    for tag in tags:
        arrays[tag] =arrays[tag].values[start:end]
    return arrays

def extract_indices(mask,kernel,future,arraysize,indices,arrays):
    counter = 0
    for i in range(arraysize-future):
        if mask[i]==True:
            counter = counter + 1
            if counter >= kernel and mask[i+future]==True:
                indices.append(list(range(i-kernel+1,i+1))+[i+future])
        else:
            counter = 0
    return indices
def extract_indices2(kernel,future,indices,arrays):
    counter = 0
    for i in range(kernel-1,len(arrays[0])-future):
        indices.append(list(range(i-kernel+1,i+1))+[i+future])
    return indices

@nb.jit
def extract_samples(array,pdt_index,kernel,indices,dataX,datay,ranger,dim):
    if dim == pdt_index:
        for i in ranger:
            r = indices[i]
            dataX[i,0:kernel,dim] = array[r[:-1]]
            datay[i] = array[r[-1]]
    else:
        for i in ranger:
            r = indices[i]
            dataX[i,0:kernel,dim] = array[r[:-1]]

    return dataX,datay
@nb.jit
def extract_samples2(dim,kernel,indices,dataX,datay,ranger,array):
    #this boy deals with multipredicative learning
    for i in ranger:
        r = indices[i]
        dataX[i,0:kernel,dim] = array[r[:-1]]
        datay[i,dim] = array[r[-1]]
    return dataX,datay

def most_freq_specific_tags(units,tags,arrays,path,time_interval):
    for u in range(len(units)):
        unit=units[u]
        best_size = 0
        for tag in tags:
            size = os.stat(path+tag+'.pickle').st_size
            if size>best_size and tag[4:6] == unit:
                best_size = size
                best_tag = tag

        arrays[best_tag] = pickle_load(path+best_tag+'.pickle')
        arrays[best_tag]=arrays[best_tag].resample('%ds'%time_interval).mean()
        arrays[best_tag]=arrays[best_tag].fillna(method='ffill')
    return arrays

def create_data_permutation(n_units,units,tags):
    permutation = np.zeros(n_units)
    for u in range(n_units):
        unit=units[u]
        count=0
        for tag in tags:
            if tag[4:6] == unit:
                count +=1
        permutation[u] = count

    for i in range(n_units):
        permutation[i] = np.random.randint(0,permutation[i])
    return permutation
def load_permutation(n_units,units,tags,arrays,path,time_interval,permutation):
    for u in range(n_units):
        unit=units[u]
        count = 0
        if permutation[u] == -1:
            continue
        for tag in tags:
            #size = os.stat(path+tag+'.pickle').st_size
            if tag[4:6] == unit:
                if count==permutation[u]:
                    arrays[tag] = pickle_load(path+tag+'.pickle')
                    arrays[tag]=arrays[tag].resample('%ds'%time_interval).mean()
                    arrays[tag]=arrays[tag].fillna(method='ffill')
                    break
                else:
                    count +=1
    return arrays
def best_permutation(filename='model_benchmarks.txt'):
    with open(filename,'r') as f:
        columns = f.readline().split(' ')
        mse = []
        median_mse = []
        permutation = []
        for line in f:
            mse.append(line.split(' ')[0])
            permutation.append(line.split(' ')[2])
    iD = np.argmin(mse)
    best_mse = mse[iD]
    best_perm = permutation[iD]
    return best_perm

def n_best_permutations(n,filename):
    with open(filename,'r') as f:
        columns = f.readline().split(' ')
        mse = []
        median_mse = []
        permutation = []
        for line in f:
            mse.append(line.split(' ')[0])
            permutation.append(line.split(' ')[2])
    ids = np.argsort(mse)[:n]
    perms = np.asarray(permutation)[np.argsort(mse)[:n]]
    mses = np.asarray(mse)[ids]
    return np.vstack((perms,mses))

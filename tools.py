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
    print(len(tags))
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
def load_tags_of_place(path,place):
    #loads tags of entire district if you've run pickle_pickler.py for that district
    tags = []
    with open(path+"%s_pickles/tags.txt"%place,"r") as f:
        for line in f:
            tags.append(line[:-1])
    return(tags)
def load_units_of_place(path,place):
    #loads tags of entire district if you've run pickle_pickler.py for that district
    tags = []
    with open(path+"%s_pickles/units.txt"%place,"r") as f:
        for line in f:
            tags.append(line[:-1])
    return(tags)

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

def tags_of_place_sparse(df,min_obs=0):
    ''''returns tags of the place you want, with a minimum amount of observations'''
    k = Counter(df['tag']) #dict object that with the # of observation of each tag
    tags = list()
    for tag in k:
        unit = tag[4:7]
        if ('vY' in tag[-3:]) and (min_obs<k[tag]) and ((unit[:2]=="PT")or(unit[:2]=="TT")or(unit=="PDT")):
            tags.append(str(tag))
    del k
    return tags

def extract_arrays_asterix(arrays,tags,df,time_interval): #this was created to deal with fragmented pickle loads that needed fragmented processing
    '''return values in the dataframe which belongs to the tags'''
    #arrays = dict()
    #df = df.sort_values(by='tag')
    already = list(arrays.keys())
    grp = df.groupby('tag',sort=False, as_index=False)
    del df
    for tag,slicee in grp:
        slicee = slicee.drop(columns=['tag'])
        slicee['Date'] = pd.to_datetime(slicee['Date'])
        slicee = slicee.set_index('Date')
        slicee = slicee.sort_index()
        slicee = slicee.resample('%ds'%time_interval).fillna(method='ffill')
        if tag in already:
            arrays[tag] = pd.concat([arrays[tag],slicee])
        else:
            arrays[tag] = slicee
    del grp,slicee
    return arrays


def extract_arrays2(tags,df,time_interval):
    '''return values in the dataframe which belongs to the tags'''
    arrays = dict()
    #df = df.sort_values(by='tag')
    grp = df.groupby('tag',sort=False, as_index=False)
    del df
    for tag,slicee in grp:
        if tag in tags:
            slicee = slicee.drop(columns=['tag'])
            slicee['Date'] = pd.to_datetime(slicee['Date'])
            slicee = slicee.set_index('Date')
            slicee = slicee.sort_index()
            slicee = slicee.resample('%ds'%time_interval).fillna(method='ffill')
            arrays[tag] = slicee
    del grp,slicee
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
import matplotlib.pyplot as plt

def create_time_variables(start,end,time_interval): #takes datetime interval, returns arrays which showcase which month/day/week/hour it currently is
    dates = pd.date_range(start, end,freq=str(time_interval)+"s") #freq=str(freq)+"s")
    timedata = dict()
    timedata["month"] = dates.month
    timedata["week"] = dates.week
    timedata["day"] = dates.day
    timedata["hour"] = dates.hour
    timedata["minute"] = dates.minute
    timedata["weekday"] = dates.weekday
    return timedata
def noncategorical_timedata(timedata):
    months = timedata["month"]
    week = timedata["week"]
    day = timedata["day"]
    hour = timedata["hour"]
    minute = timedata["minute"]
    weekday=timedata["weekday"]
    hour += minute/max(minute)
    weekday += hour/max(hour)
    day += hour/max(hour)
    week += weekday/max(weekday)

    NonCatTime = dict()
    dayperiod = day/32*2*np.pi #period of month
    minperiod = minute/60*2*np.pi #period of hour
    hourperiod = 1/24*hour*2*np.pi #period of day
    weekdayperiod = weekday/7*2*np.pi #period of week
    weekperiod = 1/52*week*2*np.pi #period of year
    #NonCatTime["sinday"] = np.sin(dayperiod)
    #NonCatTime["cosday"] = np.cos(dayperiod)
    #NonCatTime["sinweekday"] = np.sin(weekdayperiod) #actually werks
    #NonCatTime["cosweekday"] = np.cos(weekdayperiod)
    #NonCatTime["sinminute"] = np.sin(minperiod)#werks
    NonCatTime["cosminute"] = np.cos(minperiod)
    #NonCatTime["coshour"] = np.cos(hourperiod)#werks
    #NonCatTime["sinhour"] = np.sin(hourperiod)
    #NonCatTime["cosweek"] = np.cos(1/52*week*2*np.pi)
    #NonCatTime["sinweek"] = np.sin(1/52*week*2*np.pi)
    return NonCatTime
def onehot_creator(array):
    if len(array.shape) > 1:
        print("error in def onehot(array) in tools.py")
        exit()
    unique = array.unique()
    unique = sorted(unique)
    onehot = (unique == array[:,None])
    return onehot
def categorical_timedata(timedata,keys):
    #months = timedata["month"]
    #week = timedata["week"]
    #day = timedata["day"]
    #hour = timedata["hour"]
    #minute = timedata["minute"]
    #weekday=timedata["weekday"]
    onehot = 0
    for key in keys:
        if isinstance(onehot,int):
            onehot = onehot_creator(timedata[key])
        else:
            onehot = np.concatenate((onehot,onehot_creator(timedata[key])),axis=1)
    return onehot.astype(np.float64)




def remove_time_aspect(arrays,start,end):
    tags =(list(arrays.keys()))
    for tag in tags:
        arrays[tag] =arrays[tag][start:end].values
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
        arrays[best_tag]=arrays[best_tag].resample('%ds'%time_interval)
        arrays[best_tag]=arrays[best_tag].fillna(method='ffill')
    return arrays

def create_data_permutation(n_units,units,tags):
    permutation = np.zeros(n_units,dtype=np.int8)
    tags_of_unit = dict()
    for u in range(n_units):
        unit=units[u]
        count=0
        tags_of_unit[unit] = []
        for tag in tags:
            unitt = tag[4:(4+len(unit))]
            if unit == unitt:
                count +=1
                tags_of_unit[unit].append(tag)
        permutation[u] = count

    for i in range(n_units):
        rand = np.random.randint(0,permutation[i])
        permutation[i] = rand
    return permutation,tags_of_unit
def load_permutation(n_units,units,tags,arrays,path,time_interval,permutation):
    for u in range(n_units):
        unit=units[u]
        count = 0
        if permutation[u] == -1:
            continue
        for tag in tags:
            #size = os.stat(path+tag+'.pickle').st_size
            unitt = tag[4:7]
            if unitt[-1].isdigit():
                unitt = unitt[:-1]
            if unitt == unit:
                if count==permutation[u]:
                    arrays[tag] = pickle_load(path+tag+'.pickle')
                    arrays[tag]=arrays[tag].resample('%ds'%time_interval).fillna(method='ffill')
                    #arrays[tag]=arrays[tag].fillna(method='ffill')
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

def differencial_error_check(length,timesteps,future_vision,pdt_original,pdt_index,k,physical_error,stds,means,dims):
    indices = extract_indices2(timesteps,future_vision,[],[pdt_original])
    indices = np.asarray(indices)
    dataX = np.zeros((len(indices),timesteps,len(dims)),dtype=np.float)
    dataY = np.zeros((len(indices),len(dims)),dtype=np.float32)
    ranger = np.arange(len(indices))
    print("what")
    extract_samples2(pdt_index,timesteps,indices,dataX,dataY,ranger,pdt_original.squeeze())
    dataX = dataX[:,-1,pdt_index]
    dataY = dataY[:,pdt_index]
    dataY = dataY.squeeze()
    print(dataX.shape,dataY.shape)
    dataY = dataY[(dataY.shape[0]-length):]
    dataX = dataX[(dataX.shape[0]-length):]
    copy_last = (np.mean(np.abs(dataY-dataX)))

    manual_physical = (np.mean(np.abs(dataY-(dataX+k*stds[pdt_index]+means[pdt_index]))))
    print(manual_physical)
    if abs(manual_physical-physical_error) < 1e-4:
        print("error checks out")
    else:
        print("error doesn't check out: ", abs(manual_physical-physical_error))
        print(timesteps," ",future_vision)
        exit()
    return dataX, dataY, copy_last, manual_physical
def error_check(length,timesteps,future_vision,pdt_original,pdt_index,k,physical_error,stds,means,dims):
    indices = extract_indices2(timesteps,future_vision,[],[pdt_original])
    indices = np.asarray(indices)
    dataX = np.zeros((len(indices),timesteps,len(dims)),dtype=np.float)
    dataY = np.zeros((len(indices),len(dims)),dtype=np.float32)
    ranger = np.arange(len(indices))
    extract_samples2(pdt_index,timesteps,indices,dataX,dataY,ranger,pdt_original.squeeze())
    dataX = dataX[:,-1,pdt_index]
    dataY = dataY[:,pdt_index]
    dataY = dataY.squeeze()
    dataY = dataY[(dataY.shape[0]-length):]
    dataX = dataX[(dataX.shape[0]-length):]
    copy_last = (np.mean(np.abs(dataY-dataX)))
    #with open('mediancopy.txt','a')as f:
    #    f.write("%f \n" % np.median(np.abs(dataY-dataX)))
    #print("swag")
    manual_physical = np.mean(np.abs(dataY-(k*stds[pdt_index]+means[pdt_index])))
    if abs(manual_physical-physical_error) < 1e-4:
        print("error checks out")
    else:
        print("error doesn't check out: ", abs(manual_physical-physical_error))
        print(timesteps," ",future_vision)
        exit()
    return dataX, dataY, copy_last, manual_physical


def make_differential_data(arrays,future_vision,tags):
    Arrays = np.zeros((len(tags),len(arrays[tags[0]])-future_vision),dtype=np.float32)
    for i in range(len(tags)):
        Arrays[i] =  arrays[tags[i]].squeeze()[future_vision:] - arrays[tags[i]].squeeze()[0:-future_vision]
    return Arrays

def make_differential_data2(arrays,future_vision,tags):
    Arrays = np.zeros((2*len(tags),len(arrays[tags[0]])-1),dtype=np.float32)
    for i in range(len(tags)):
        Arrays[i] =  arrays[tags[i]].squeeze()[:-1]
    for i in range(len(tags),2*len(tags)):
        Arrays[i] =  arrays[tags[i-len(tags)]].squeeze()[1:] - arrays[tags[i-len(tags)]].squeeze()[0:-1]
    return Arrays

from tools import *
import pandas as pd
#import matplotlib.pyplot as plt

def load_perm(time_interval,future,kernel_size,permutation,place="none",smooth=0,multipred=0,weather=0,time=0,path_to_places='none'):
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

    if multipred and permutation != 'none':

        permutations = permutation
        permutations = [int(i) for i in permutations.split(' ')]
        n_units = len(units)
        arrays = load_permutation(n_units,units,tags,arrays,path,time_interval,permutations)
        tags = list(arrays.keys())
        arrays = cross_reference_datetimes(tags,arrays)
        n_units = len(tags)
    elif place != 'none' and path!= 'none':
        path_to_places = path_to_places+place+'.zip'
        df = smart_read(path_to_places)
        #print(tags)
        #print(len(tags))
        exit()
    else:
        n_units = 1
        for tag in tags:
            if "PDT" in tag:
                arrays = dict()
                arrays[tag] =  pickle_load(path+tag+'.pickle')
                #arrays[tag]=arrays[tag].resample('s').mean()
                arrays[tag]=arrays[tag].resample('%ds'%time_interval).mean()
                arrays[tag]=arrays[tag].fillna(method='ffill')

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



    Arrays = np.zeros((n_units,len(arrays[tags[0]])-1),dtype=np.float32)


    i=0
    means0 = []
    for t in range(len(tags)):
        tag = tags[t]
        if 'PDT' in tag:
            pdt_index = i #line below calculates the differenced series
        means0.append(arrays[tag][0])
        Arrays[i] =  arrays[tag].squeeze()[1:] - arrays[tag].squeeze()[0:-1]
        #plt.plot(arrays[tag][-1007:],label='before scaling')

        i+=1
    del arrays

    dims = list(range(Arrays.shape[0]))
    means=[]
    stds=[]
    for i in dims:
        mean = Arrays[i].mean()
        std = Arrays[i].std()
        Arrays[i] -= mean
        Arrays[i] /= std
        means.append(mean)
        stds.append(std)



    #here we create non-categorical time data
    if multipred and time:
        sin,cos = create_time_variables(start,end,time_interval)
        Arrays = np.vstack((Arrays,sin[:-1]))
        Arrays = np.vstack((Arrays,cos[:-1]))


    indices = extract_indices2(kernel_size,future,[],Arrays)

    indices = np.asarray(indices)
    ranger = np.arange(len(indices))
    dataX = np.zeros((len(indices),kernel_size,Arrays.shape[0]),dtype=np.float)
    datay = np.zeros((len(indices),Arrays.shape[0]),dtype=np.float32)
    X = [extract_samples2(x,kernel_size,indices,dataX,datay,ranger,Arrays[x]) for x in dims]
    #X = parmap.map(extract_samples2,dims,kernel_size,indices,dataX,datay,ranger,Arrays)

    del X,Arrays,indices,ind
    return dataX, datay,pdt_index,tags[pdt_index],means0,means,stds
'''
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
'''

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
#press = 1000*(1-(Arrays[0]+288.9414)*(Arrays[0]-3.9863)**2)/508929.2*(Arrays[0]*68.12963)

#Arrays = np.vstack((Arrays,press))
#Arrays=Arrays[:,1:]-Arrays[:,0:-1]
#Arrays=(Arrays[:,1:]-Arrays[:,0:-1])/time_interval
#Arrays[1] = np.abs(Arrays[1])

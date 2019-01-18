from tools import * 
import matplotlib.pyplot as plt
def load(place,path,units,kernel_size,time_interval,future):
    a = datetime.now()
    df = pickle_load(path)   #all tags, values and timestamps     #takes two sec
    print(datetime.now()-a)

    a = datetime.now()
    tags = tags_of_place(df) #the tags in a hierarchichal fashion #takes one sec
    tags= list(tags)
    print(datetime.now()-a)

    a = datetime.now()
    df = extract_arrays2(tags,df,time_interval) 
    print(datetime.now()-a)
    if place =='VIK':
        for tag in tags:
            if "VIK_PDT2002" in tag:
                VIK_PDT = tag
                break
    a = datetime.now()
    relevant_tags = most_freq_plural(units,tags,df,time_interval)
    print(datetime.now()-a)
    if place =="VIK":
        for t in range(len(relevant_tags)):
            if "PDT" in relevant_tags[t]:
                relevant_tags[t] = VIK_PDT
    for tag in tags:
        if tag not in relevant_tags:
            del df[tag]
        else:
            df[tag] = df[tag].resample('%ds'%time_interval).mean()
            df[tag].fillna(method='ffill',inplace=True)
            df[tag].fillna(method='bfill',inplace=True)

    a = datetime.now()
    arrays = cross_reference_datetimes(relevant_tags,df)
    print(datetime.now()-a)
    a = datetime.now()
    arrays = remove_time_aspect(df)
    print(datetime.now()-a)
    del df 
    Arrays = np.zeros((len(arrays),len(arrays[relevant_tags[0]])),dtype=np.float32)
    i=0
    for t in range(len(relevant_tags)):
        tag = relevant_tags[t]
        if 'PDT' in tag:
            pdt_index = i
        Arrays[i] = arrays[tag].squeeze()
        i+=1
    del arrays

    a = datetime.now()
    indices = extract_indices2(kernel_size,future,[],Arrays) 
    print(datetime.now()-a)

    dataX = np.zeros((len(indices),kernel_size,len(relevant_tags)),dtype=np.float)
    datay = np.zeros((len(indices),1),dtype=np.float32)
    a = datetime.now()
    ranger = np.arange(len(indices))
    dims = len(relevant_tags)
    #dataX,datay = extract_samples(pdt_index,Arrays,kernel_size,indices,dataX,datay,ranger,dims) 
    import parmap
    X = [extract_samples(Arrays[x],pdt_index,kernel_size,indices,dataX,datay,ranger,x) for x in range(dims)]
    del X
    print(datetime.now()-a)

    dataX_mean = np.mean(dataX,axis=(1,0)).squeeze()

    datay_mean = np.mean(datay)

    dataX_std = np.std(dataX,axis=(1,0)).squeeze()
    datay_std = np.std(datay)
    del Arrays,indices
    dataX = (dataX-dataX_mean)/dataX_std
    datay = (datay-datay_mean)/datay_std
    return dataX, datay,pdt_index
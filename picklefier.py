'''takes a district's data, filters out lazy tags and picklefies it'''
import pandas as pd
from datetime import datetime
from tools import tags_of_place,pickle_load, smart_read
places = []
file = open('/home/josephkn/Documents/Fortum/master/weatherdata/area_codes.txt','r')
for line in file:
    p = line[:-1]
    places.append(p)
file.close()
n = 3
for i in range(n-1,n):
    place = places[i]
    a = datetime.now()
    filename = '/home/josephkn/Documents/Fortum/master/weatherdata/%s'%place
    df = smart_read(filename+'.zip') #takes all tags from a location and keeps the most frequent
    df.to_pickle('%s6.pickle'%place) #4 means we only allow for sensors with more than 50k reading over the year
                                    #5 keeps those over 25k measurements
                                    #6 keeps those over 20k
    del df
    print(place)
    print(datetime.now()-a)
    #except:
    print("rip %s"%place)

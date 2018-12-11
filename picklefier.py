'''takes a district's data, filters out lazy tags and picklefies it'''
import pandas as pd 
from datetime import datetime
from tools import tags_of_place,pickle_load, smart_read
'''
places = []
file = open('/home/josephkn/Documents/Fortum/weatherdata/area_codes.txt','r')
for line in file:
    p = line[:-1]
    places.append(p)
file.close()
n = 17
for i in range(n):
    place = places[i]
'''
#try:
place = "VIK"
a = datetime.now()
filename = '%s'%place
#nrows = int(40000) #how many lines to read at a time
df = smart_read(filename)
df.to_pickle('%s5.pickle'%place) #4 means we only allow for sensors with more than 50k reading over the year
del df
print(place)
print(datetime.now()-a)
#except:
#        print("rip")
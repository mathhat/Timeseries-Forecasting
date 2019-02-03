#from utilize import fetch , multi_pred
from perceptron import train
import matplotlib.pyplot as plt

permutation = "0 0 0 0 0 0 0 0" #['TT', 'JY', 'TY', 'FT', 'JT', 'TZ', 'PD', 'PR']
multipred=0 #wanna look at lotsa features, or just PDT?
placepred = 1 #wanna do place based, or special subset of sensors?
place = 'VIK' #if you wanna do place based, which place?
differentiate = 0
'''
places = []
file = open('/home/josephkn/Documents/Fortum/master/weatherdata/area_codes.txt','r')
for line in file:
    p = line[:-1]
    places.append(p)
places = places[:-1]
'''
path_to_places = "/home/josephkn/Documents/Fortum/master/weatherdata/"
timesteps = 10; future = 1; time_interval = 180
activation = 'relu'; epochs=5; nodes=128;batch_size=64;
weather = 0 #multipred must also be true for weatherdata to be used in model
time = 0
runs = 5
if place!='none':
    permutation = 'none'
for differentiate in [0,1]:
    for time_interval in [180,300,600,1200]:
        for timesteps in [2,5,10,15]:
            for run in range(runs):
                #try:
                model, testx, testy,pdt_index = train( #trains model based on the variable above
                    timesteps,future,time_interval,Permutation=permutation,place=place,activation=activation,
                    preserve=0,
                    patience=0,
                    smooth=0,
                    epochs=epochs,
                    multipred=multipred,
                    nodes1=nodes,
                    nodes2=nodes,
                    batch_size=batch_size,
                    weather=weather,
                    time=time,
                    path=path_to_places,
                    differentiate=differentiate)
                '''
                except:
                    with open("bugs.txt","a") as file:
                        for i in permutation:
                            file.write("%d,"%i)
                        file.write("\n")
'''

#from utilize import fetch , multi_pred
from model import train
import matplotlib.pyplot as plt

multipred=1 #wanna look at lotsa features, or just PDT?
placepred = 1 #wanna do place based, or special subset of sensors?
place = 'VIK' #if you wanna do place based, which place?
'''
places = []
file = open('/home/josephkn/Documents/Fortum/master/weatherdata/area_codes.txt','r')
for line in file:
    p = line[:-1]
    places.append(p)
places = places[:-1]
'''
path_to_places = "/home/josephkn/Documents/Fortum/master/weatherdata/"
placepred = 1 #switch that activates place based model
timesteps = 10; future = 1; time_interval = 300
activation = 'sigmoid'; epochs=5; nodes=128;batch_size=64;
weather = 0 #multipred must also be true for weatherdata to be used in model
time = 0
runs = 100
permutations = []
path ="/home/josephkn/Documents/Fortum/master/bench_place/"
with open('/home/josephkn/Documents/Fortum/master/bench_place/benchmarks_%s_%dinterval_%dsteps_multipred.txt'%(place,time_interval,timesteps),'r') as f:
    hyperparameters = f.readline().split(" ")
    print(hyperparameters)
    index = hyperparameters.index("permutation")
    for i in range(runs):
        line = f.readline().split(" ")
        permutations.append([int(i) for i in line[index].split(",")])
for run in range(runs):
    try:
        permutation = permutations[run]
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
            path=path_to_places)
    except:
        with open("bugs.txt","a") as file:
            for i in permutation:
                file.write("%d,"%i)
            file.write("\n")

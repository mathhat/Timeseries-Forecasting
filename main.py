#from utilize import fetch , multi_pred
from model import train
import matplotlib.pyplot as plt

permutation = "0 0 0 0 0 0 0 0" #['TT', 'JY', 'TY', 'FT', 'JT', 'TZ', 'PD', 'PR'] #loads Oda's tags
optimized_features = 1 #wanna choose features based on a linear or mutual information?

method = "covar" #which method? "MIR", "covar", "f_score"
how_many = 10 #how many features would you like?
special = 0 #want special features?
n_special = 5 #how many?
if optimized_features==0:#incase you retrded
    special=0
multipred=1 #wanna look at lotsa features, or just PDT?
placepred = 1 #wanna do place based, or special subset of sensors?
place = 'VIK' #if you wanna do place based, which place?
differentiate = 1
'''
places = []
file = open('/home/josephkn/Documents/Fortum/master/weatherdata/area_codes.txt','r')
for line in file:
    p = line[:-1]
    places.append(p)
places = places[:-1]
'''
timesteps = 20; future = 10; time_interval = 60
activation = 'relu'; epochs=10; nodes=128;batch_size=64;
weather = 1 #multipred and optimized feats must also be true for weatherdata to be used in model
time = 0
categorical = 0 #
keys = ["weekday"] #which categories of time do you want to add
holiday = 1 #adding holidays to your categorical time data
if categorical ==0:
    holiday=0 #in case you messed up
runs = 5
if place!='none':
    permutation = 'none'
ii = 0
'''
for activation in ["relu","sigmoid"]:
    for timesteps in [2,5,20,25,30]:
        for time_interval in [60,120]:
            for differentiate in[0,1]:
                for future in [1,2,3,5,10,15,20,25]:
                    '''
for method in ["covar"]:
    for future in [10,15,20,25,30,35,40]:
        model, testx, testy,pdt_index = train(
            #trains model based on the variables above
            timesteps,future,time_interval,Permutation=permutation,
            place=place,activation=activation,preserve=0,
            patience=0, smooth=0, epochs=epochs, multipred=multipred,
            nodes1=nodes, nodes2=nodes, batch_size=batch_size,
            weather=weather, time=time, differentiate=differentiate,
            method =method, optimized_features=optimized_features,
            k=how_many,categorical=categorical,catkeys=keys,holiday=holiday,
            k2=n_special,special=special)

'''
                except:
                    with open("bugs.txt","a") as file:
                        for i in permutation:
                            file.write("%d,"%i)
                        file.write("\n")
'''

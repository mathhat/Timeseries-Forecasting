#from utilize import fetch , multi_pred
from perceptron import train
import matplotlib.pyplot as plt

permutation = "0 0 0 0 0 0 0 0" #['TT', 'JY', 'TY', 'FT', 'JT', 'TZ', 'PD', 'PR'] #loads Oda's tags
optimized_features = 1 #wanna choose features based on a linear or mutual information?
method = "MIR" #which method? "MIR", "covar", "f_score"
how_many = 20 #how many features would you like?

multipred=0 #wanna look at lotsa features, or just PDT?
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
timesteps = 20; future = 2; time_interval = 60
activation = 'relu'; epochs=20; nodes=128;batch_size=64;
weather = 0 #multipred must also be true for weatherdata to be used in model
time = 0
runs = 5
if place!='none':
    permutation = 'none'
ii = 0
for activation in ["relu","sigmoid"]:
    for timesteps in [2,5,20,25,30]:
        for time_interval in [60,120]:
            for differentiate in[0,1]:
                for future in [1,2,3,5,10,15,20,25]:
                    model, testx, testy,pdt_index = train(
                        #trains model based on the variables above
                        timesteps,future,time_interval,Permutation=permutation,
                        place=place,activation=activation,preserve=0,
                        patience=0, smooth=0, epochs=epochs, multipred=multipred,
                        nodes1=nodes, nodes2=nodes, batch_size=batch_size,
                        weather=weather, time=time, differentiate=differentiate)#,
                        #method =method, optimized_features=optimized_features,
                        #k=how_many)

'''
                except:
                    with open("bugs.txt","a") as file:
                        for i in permutation:
                            file.write("%d,"%i)
                        file.write("\n")
'''

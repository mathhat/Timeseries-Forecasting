from subprocess import call
places = []
print("check the amount of lines left and fill into variable n")
#exit()
#file = open('area_codes_no_bjo.txt','r')
#for line in file:
#    p = line[:-1]
#    places.append(p)
#file.close()

#n = 6
#for i in range(n):
#    place = places[i]
place = "TOY"
file = open('%s.txt'%place,'w')
file.close()
call("grep \"%s\" /home/josephkn/Documents/fov_data_v1 > %s.txt"%(place,place),shell=True)

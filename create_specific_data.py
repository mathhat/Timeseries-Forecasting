from subprocess import call
with open('tags.txt','r') as f:
    tags = f.readline().split(' ')
    tags[-1] = tags[-1][:-1]

    for tag in tags:
        place = tag[:3]
        call("grep \"%s\" /home/josephkn/Documents/Fortum/weatherdata/%s.txt > taglist/%s"%(tag,place,tag),shell=True)
        call('wc -l taglist/%s'%tag,shell=True)
import numpy as np
import matplotlib.pyplot as plt
def parse(dirpath,timesteps,interval,differentiated):
    m = []
    with open('%s/benchmarks_%dinterval_unipred.txt'%(dirpath,interval),'r') as f:
        args = f.readline().split(',')
        try:
            diff = args.index('differentiate')
        except:
            diff= args.index('differentiate\n')
        steps = args.index('timesteps')
        for line in f:
            line = line.split(' ')
            if int(line[diff][0]) == differentiated and int(line[steps]) == timesteps:
                m.append(float(line[0]))
    return m
Timesteps = [2,5,10,15]
Differentiate = [0,1]
Interval = [60,180,300,600,1200]
timesteps = 10
differentiate = 0
interval = 1200
#m = parse('bench_uni',timesteps,interval,differentiate)
#print(np.mean(m))
#m = parse('bench_uni_lstm_dense',timesteps,interval,differentiate)
#print(np.mean(m))
#m = parse('bench_uni_lstm',timesteps,interval,differentiate)
#print(np.mean(m))
#exit()
plt.style.use(['ggplot','Solarize_Light2','bmh'])

diff = []
non_diff=[]
for differentiate in Differentiate:
    for interval in Interval:
        lstm=[]
        lstmd=[]
        perceptron=[]
        for timesteps in Timesteps:
            lstm.append(parse('bench_uni_lstm',timesteps,interval,differentiate))
            lstmd.append(parse('bench_uni_lstm_dense',timesteps,interval,differentiate))
            perceptron.append(parse('bench_uni',timesteps,interval,differentiate))
        print(interval,' interval')
        print(np.mean(lstm),' ',differentiate,' lstm')
        print(np.mean(lstmd),' ',differentiate,' lstmd')
        print(np.mean(perceptron),' ',differentiate,' perceptron')

'''
    plt.title('Error for Models When dt = %s'%interval)#,size=22)
    plt.xlabel('N timesteps')#,size=18)
    plt.ylabel('Error [kPa]')#,size=18)
    plt.plot(Timesteps,lstm,'r-^',label='lstm')
    plt.plot(Timesteps,lstmd,'b-o',label='lstmd')
    plt.plot(Timesteps,perceptron,'y-*',label='percpetron')
    plt.legend()
    plt.show()
'''

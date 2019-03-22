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
    return np.mean(m)
Timesteps = [2,5,10,15]
Differentiate = [0,1]
Interval = [60,180,300,600,1200]
timesteps = 10
#differentiate = 0
#interval = 1200
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
        lstm2=[]
        lstmd2=[]
        perceptron2=[]
        for timesteps in Timesteps:
            lstm2.append(parse('bench_uni_lstm2',timesteps,interval,differentiate))
            #print(parse('bench_uni_lstm2',timesteps,interval,differentiate))
            lstmd2.append(parse('bench_uni_lstm_dense2',timesteps,interval,differentiate))
            perceptron2.append(parse('bench_uni2',timesteps,interval,differentiate))
            lstm.append(parse('bench_uni_lstm',timesteps,interval,differentiate))
            lstmd.append(parse('bench_uni_lstm_dense',timesteps,interval,differentiate))
            perceptron.append(parse('bench_uni',timesteps,interval,differentiate))
        print(interval,' interval')
        print(np.mean(lstm),' ',differentiate,' lstm')
        print(np.mean(lstmd),' ',differentiate,' lstmd')
        print(np.mean(perceptron),' ',differentiate,' perceptron')
        print(np.mean(lstm2),' ',differentiate,' lstm2')
        print(np.mean(lstmd2),' ',differentiate,' lstmd2')
        print(np.mean(perceptron2),' ',differentiate,' perceptron2')

        if differentiate:
            k='Differentiated'
        else:
            k=''
        plt.title('Error for Models When dt = %s %s'%(interval,k))#,size=22)
        plt.xlabel('N timesteps')#,size=18)
        plt.ylabel('Error [kPa]')#,size=18)
        plt.plot(Timesteps,lstm2,'r-^',label='lstm_sig')
        plt.plot(Timesteps,lstmd2,'b-o',label='lstm_dense_sig')
        plt.plot(Timesteps,perceptron2,'y-*',label='perceptron_sig')

        plt.plot(Timesteps,lstm,'c-^',label='lstm_relu')
        plt.plot(Timesteps,lstmd,'b-o',label='lstm_dense_relu')
        plt.plot(Timesteps,perceptron,'k-*',label='perceptron_relu')
        plt.xlim(1,16)
        plt.ylim(0,int(1.5*max(lstm+lstm2+lstmd+lstmd2+perceptron+perceptron2)))
        plt.legend()
        plt.show()

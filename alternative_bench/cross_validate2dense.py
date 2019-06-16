import numpy as np
import matplotlib.pyplot as plt
from cross_validate2 import crosslstm
from multi_lstm_bench import lstm_multi
import sys
sys.path.insert(0,'../')
from linear_regression import linear_results

def crosslstmd(activation="relu"):
    def parse(dirpath,timesteps,interval,differentiated,future_vision,copylast=0,median=0):
        m = []
        c = []
        med = []
        with open('%s/benchmarks_%dinterval_unipred.txt'%(dirpath,interval),'r') as f:
            args = f.readline().split(',')
            try:
                diff = args.index('differentiate')
            except:
                diff= args.index('differentiate\n')
            steps = args.index('timesteps')
            future = args.index('future_vision')
            for line in f:
                line = line.split(' ')
                if int(line[diff][0]) == differentiated and int(line[steps]) == timesteps and int(line[future]) == future_vision:
                    m.append(float(line[0]))
                    if copylast:
                        c.append(float(line[3]))
                    if median:
                        med.append(float(line[2]))
        if copylast and median:
            if median:
                return np.mean(m), np.mean(c),np.mean(med)
        elif copylast:
            return np.mean(m), np.mean(c)
        elif median:
            return np.mean(m), np.mean(med)
        return np.mean(m)
    Timesteps = [20,25,30]
    Differentiate = [0,1]
    Interval = [60,120,180]
    Future_vision = [1,2,3,5,10,15,20,25]
    timesteps = 10

    plt.style.use(['ggplot','Solarize_Light2','bmh'])

    diff = []
    non_diff=[]
    intervals = 3
    cols = 2

    fig, axs =  plt.subplots(intervals,cols)
    axs = axs.ravel()

    symbols = dict()
    for symbol,interval in zip(['o','^','*','x','s','p',"1"],Timesteps):
        symbols[interval] = symbol
    copy_last = dict()
    error_table = dict()
    for differentiate in [0,1]:
        for timesteps in Timesteps:
            i = 0
            for interval in [60,120,180]:#,300,600]:
                lstm=[]
                lstmd=[]
                perceptron=[]
                lstm2=[]
                lstmd2=[]
                perceptron2=[]
                j=0
                for future_vision in Future_vision:
                    if activation=="relu":
                        m,c = parse('lstm_dense',timesteps,interval,differentiate,future_vision,copylast=1)
                    else:
                        m,c = parse('lstm_dense2',timesteps,interval,differentiate,future_vision,copylast=1)
                    lstm.append(m)
                    lstm2.append(c)
                    copy_last[future_vision*interval]=c
                    try:
                        error_table[int(future_vision*interval)].append([lstm[j],interval,timesteps,differentiate])
                    except:
                        error_table[int(future_vision*interval)] = [[lstm[j],interval,timesteps,differentiate]]
                    j+=1

                #print(interval,' interval')
                #print("  ", lstm, "  ")
                if differentiate:
                    k='Diff'
                else:
                    k=''

                symb = symbols[timesteps]
                id = i*cols

                if differentiate:
                    id += 1

                axs[id].plot(np.asarray(Future_vision)*interval/60,lstm,'-'+symb,label='POV %d'%(timesteps),alpha=0.5)
                if timesteps == Timesteps[0]:
                    axs[id].plot(np.asarray(Future_vision)*interval/60,lstm2,'k-', label ='copy last')
                axs[id].legend(loc=4)
                axs[id].set_title("dt = %d %s"%(interval,k))

                axs[id].set_xlim([0, 40])
                axs[id].set_ylim([0, 50])
                if id >= cols*intervals-2:
                    axs[id].set_xlabel("Foresight [Minutes]")

                if id%2==0:
                    axs[id].set_ylabel("Prediction Error [kPa]")

                i+=1

                #plt.plot(Timesteps,lstmd,'b-o',label='lstm_dense_relu')
                #plt.plot(Timesteps,perceptron,'k-*',label='perceptron_relu')
    if activation=="relu":
        plt.suptitle("Error for Univariate Forecasting Models, LSTM+FF(ReLU)",size=20)
    else:
        plt.suptitle("Error for Univariate Forecasting Models, LSTM+FF(Sigmoid)",size=20)
    plt.show()

    n = len(error_table.keys())
    contests = dict()
    results = dict()
    keys = list(error_table.keys())
    minerrors = dict()
    for i in range(n):
        forecast = keys[i]
        #print(error_table[i])
        best = np.inf
        table = np.asarray(error_table[forecast])
        contests[forecast] = np.unique(table[:,1])
        errs = table[:,0] #separate errors from each result triplet to find the best triplet
        id = np.argmin(errs)
        #print(errs)

        minerrors[forecast] = errs[id]
        results[forecast]=table[id]
    best = []
    time = []
    cp = []
    for i in range(n):
        forecast = sorted(keys)[i]
        if forecast/60 > 40:
            continue
        if len(contests[forecast])>1:
            print("for forecasting ", forecast/60," minute(s) into the future, at dt of",
                    results[forecast][1]," and ", results[forecast][2], "timesteps of performs best",contests[forecast], results[forecast][0]," ",results[forecast][-1])
        best.append(minerrors[forecast])
        time.append(forecast/60)
        cp.append(copy_last[forecast])
    '''
    plt.plot(time,cp,label="f(x)=x")
    plt.plot(time,best,label="best univariate results")
    plt.plot(time)
    plt.legend()
    plt.show()
    '''
    return cp, time, best,results
print("lstm + sigmoid")
cp, time, bestlstmdsig,resultsig = crosslstmd(activation="sigmoid")
print("lstm + relu")
cp, time, bestlstmdrelu,resultrel = crosslstmd(activation="relu")
print("lstm ")
tt,lstmmulti =lstm_multi()
time2, bestlstm,result = crosslstm()
lin,time3,cps,t = linear_results()
plt.title("Univariate")
plt.plot(time,cp,"k-o",label="copy_last")
plt.plot(time,bestlstmdrelu,"c-*",label="LSTM relu")
plt.plot(time,bestlstmdsig,"b-s",label="LSTM sig")
plt.plot(time2,bestlstm,"r-^",label="LSTM")
plt.plot(time3,lin,"m-o",label="linear regression")
plt.xlabel("N Minutes into Future")

plt.plot(tt,lstmmulti,"r->",label="LSTM multivariate")

plt.legend()
plt.show()

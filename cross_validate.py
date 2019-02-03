import numpy as np
def parse(dirpath,timesteps,interval,differentiated):
    m = []
    with open('%s/benchmarks_%dinterval_unipred.txt'%(dirpath,interval),'r') as f:
        args = f.readline().split(',')
        diff = args.index('differentiate')
        steps = args.index('timesteps')
        for line in f:
            line = line.split(' ')
            if int(line[diff][0]) == differentiated and int(line[steps]) == timesteps:
                m.append(float(line[0]))
    return m
timesteps = 10
differentiate = 1
interval = 600
m = parse('bench_uni',timesteps,interval,differentiate)
print(np.mean(m))
m = parse('bench_uni_lstm_dense',timesteps,interval,differentiate)
print(np.mean(m))
m = parse('bench_uni_lstm',timesteps,interval,differentiate)
print(np.mean(m))

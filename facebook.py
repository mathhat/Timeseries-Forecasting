from fbprophet import Prophet
from load_multi_pred import load_perm, load_features
time_interval = 60
future_vision = 1
timesteps = 20
Permutation ="none"
place = "VIK"
arrays,tags,pdt_index,pdt_tag,permutation,start,end = load_perm(time_interval,timesteps,Permutation,place)
df = arrays[pdt_tag]
print(df)
#m = Prophet()
#m.fit(df)

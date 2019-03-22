import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(1)
n = 501
data = np.linspace(20,23,n)
time = np.linspace(0,20,n)
noise = np.random.random((n))
for n in range(len(noise)):
    if np.random.randint(2):
        noise[n] *= -1
data += noise
intercepts = np.linspace(10,30,n)
slopes = np.linspace(3/20-6/20,3/20+6/20,n)
def run(inter,slope):
    model = time*slope + inter
    return np.sum((data-model)**2)
errors = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        errors[i,j] = run(intercepts[i],slopes[j])
analslope = (sum(data*time)-n*np.mean(data)*np.mean(time))/(sum(time*time)-n*np.mean(time)**2)
analinter = np.mean(data) - analslope * np.mean(time)
print(run(analinter,analslope))
print(np.min(errors))


plt.ylabel('Intercept')
g=sns.heatmap(errors,cmap="coolwarm_r",xticklabels=0,yticklabels=0)
g.set_title(r'SSR($\beta$,$\beta_0$) Space for $\hat{T}$',size=16)
g.set_xlabel(r'$\beta$',size=15)
g.set_ylabel(r'$\beta_0$',size=15)

plt.show()

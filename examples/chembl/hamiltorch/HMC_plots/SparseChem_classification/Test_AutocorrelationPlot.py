import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import sparsechem as sc
import torch.optim as optim


####################################################################################################################

# Creating Autocorrelation plot
'''x_plot = pd.plotting.autocorrelation_plot(x)
 
# plotting the Curve
x_plot.plot()
'''
'''#plt.xlim((0, x.shape[0]))
coeff_list=[]
i=0
for i in range(x.shape[0]):
    print('i------------', i)
    print(x[[0,i]])
    coeff_list.append(np.corrcoef(x[[0,i]]))
    i+=1
print(coeff_list)

#plt.savefig('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/SparseChem_classification/Target1482/Test.png')

def autocorr(y, lag):
    print(y.shape)
    
    Calculates autocorrelation coefficient for single lag value
        
    y: array
       Input time series array
    lag: int, default: 2 
         'kth' lag value
        
    Returns
    int: autocorrelation coefficient 
    
    
    y_bar =np.sum(y)/y.shape[0] #y_bar = mean of the time series y
    #print('1',y_bar)
    denominator = sum((y - y_bar) ** 2) #sum of squared differences between y(t) and y_bar
    print('2', denominator)
    numerator_p1 = y[lag:] - y_bar #y(t+k)-y_bar: difference between time series (from 'lag' till the end) and y_bar
    #print('3', numerator_p1)
    numerator_p2 = y[:len(x)-lag] - y_bar #y(t)-y_bar: difference between time series (from the start till lag) and y_bar
    #print('4', numerator_p2)
    #print(len(numerator_p1), len(numerator_p2))
    numerator = sum(numerator_p1 * numerator_p2) #sum of y(t)-y_bar and y(t-k)-y_bar
    print('5', numerator)
    return numerator/denominator

autocorr_list=[]
for i in range(0, x.shape[0]-1):
    print('i--------', i)
    autocorr_list.append(autocorr(x, i))

print(autocorr_list)

plt.plot(range(1, x.shape[0]),autocorr_list, marker='o')
plt.xlabel('lag')
plt.ylabel('autocorrelation')
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/Test.png')'''

#Autocorrelation with Pandas
params=np.load('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/SparseChem_classification/Target1482/Params_stepSize1e-05_numSteps1000_burnIn0_numSamples100_hiddenSize5_numTraining5595_tauOut0.5_tauList5.0.npy')
index=np.random.choice(params.shape[1], 10, replace=False)
print(params.shape)

for l in range(index.shape[0]):
    params_now=params[:, index[l]]
    print(params_now)
    x=pd.plotting.autocorrelation_plot(params_now)
    x.plot()  
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/SparseChem_classification/Target1482/PD_Autocorr_stepSize1e-05_numSteps1000_burnIn0_numSamples100_hiddenSize5_numTraining5595_tauOut0.5_tauList5.0.png')
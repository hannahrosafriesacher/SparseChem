import numpy as np
import matplotlib.pyplot as plt

filename='Params_stepSize1e-05_numSteps800_burnIn80_numSamples200_hiddenSize5_numTraining5595_tauOut0.9_tauList5.0.npy'
file='/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/SparseChem_classification/Target1482/'+filename
#load Parameters
Params=np.load(file)
print(Params.shape)
indeces=np.random.choice(range(0,Params.shape[0]), size=1, replace=False)
'''for i in range(indeces.shape[0]):
    Params_single=Params[:,indeces[i]]
    #plot Parameters
    plt.plot(Params_single)'''

for k in range(Params.shape[1]):
    params_now=Params[:, k]
    if params_now[0]<-0.2 and params_now[0]>-0.3:
        plt.plot(params_now)
        print(params_now)
#save plot
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/ParameterPlots_SparseChemClassification/Target1482/ParamsPlot_'+str(filename[:-4])+'.png')
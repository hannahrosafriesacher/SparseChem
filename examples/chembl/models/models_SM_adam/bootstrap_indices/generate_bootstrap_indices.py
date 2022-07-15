import numpy as np
import sparsechem as sc

x=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy')
y=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')

keep    = np.isin(folding, [2,3,4,5])
x_training = x[keep, :] 

print(x.shape, x.size)
print(x_training.shape, x_training.size)

total_number_of_samples=x_training.shape[0]
array_indeces=np.arange(total_number_of_samples)
bootstrap_list=[]
number_of_samples=100
bootstrap_sample_size=round(total_number_of_samples*1)
for i in range(0, number_of_samples):
    bootstrap_sample=np.random.choice(array_indeces, bootstrap_sample_size)
    bootstrap_list.append(bootstrap_sample)
indeces_for_bootstrapping=np.array(bootstrap_list)
print(indeces_for_bootstrapping.shape)
np.save('/home/rosa/git/SparseChem/examples/chembl/models/models_SM_adam/bootstrap_repeats/bootstrap_indices/indices_bootstrapTrainingSet_'+str(number_of_samples)+'samples_'+str(round(bootstrap_sample_size/total_number_of_samples*100))+'percSamplesize.npy', indeces_for_bootstrapping)

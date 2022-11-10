import sparsechem as sc
import numpy as np
TargetID=184
#selection of one target in the class_df
x=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy')
print(x.shape) 
y= sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
print(y.shape)
y_target=y[:, TargetID]
print(y_target.shape)
folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')
print(folding.shape)

y_nonzero_index=np.nonzero(y_target)
print(y_nonzero_index[0].shape)

y_final=y_target[y_nonzero_index[0]]
print(y_final.shape)
x_final=x[y_nonzero_index[0]]
print(x_final.shape)
folding_final=folding[y_nonzero_index[0]]
print(folding_final.shape)

np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/X_'+str(TargetID)+'_reduced.npy', x_final)
np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/y_'+str(TargetID)+'_reduced.npy', y_final)
np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/folding_'+str(TargetID)+'_reduced.npy', folding_final)




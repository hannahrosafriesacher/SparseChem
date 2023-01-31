import sparsechem as sc
import numpy as np
TargetID=[2578, 1603, 1810, 1560, 917, 3123, 1529, 686, 830, 1367, 2653, 192, 1358, 1127, 3346, 3211, 334]
for i in TargetID:
    #selection of one target in the class_df
    x=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_X.npy')
    y= sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
    y_target=y[:, i]
    folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')


    y_nonzero_index=np.nonzero(y_target)

    y_final=y_target[y_nonzero_index[0]]
    print(y_final.shape)
    x_final=x[y_nonzero_index[0]]
    folding_final=folding[y_nonzero_index[0]]

    np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/new_012023/X_'+str(i)+'_reduced.npy', x_final)
    np.save('//home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/new_012023/y_'+str(i)+'_reduced.npy', y_final)
    np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/new_012023/folding_'+str(i)+'_reduced.npy', folding_final)




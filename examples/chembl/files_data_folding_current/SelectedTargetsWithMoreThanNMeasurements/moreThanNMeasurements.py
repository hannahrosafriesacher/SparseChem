import numpy as np
import sparsechem as sc

y_class=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')

def condition(array):
    if (np.count_nonzero(array==1)>=5) and (np.count_nonzero(array==-1)>=5):
        return True
    else:
        return False

def keep_folds(fold):
    keep    = np.isin(folding, fold)
    y_class_fold = sc.keep_row_data(y_class, keep)
    y_class_fold=y_class_fold.toarray()
    return y_class_fold

def condition_for_fold(array):
    bool_list_fold=[]
    for i in range(array.shape[1]):
        #     select current target
        y_class_now=array[:,i]
        bool_list_fold.append(condition(y_class_now))
    bool_list.append(bool_list_fold)

y_class_fold0=keep_folds(0)
y_class_fold1=keep_folds(1)
y_class_fold2=keep_folds(2)
y_class_fold3=keep_folds(3)
y_class_fold4=keep_folds(4)

bool_list=[]
#keep targets with more than 5 actives/inactivaes in each fold
condition_for_fold(y_class_fold0)
condition_for_fold(y_class_fold1)
condition_for_fold(y_class_fold2)
condition_for_fold(y_class_fold3)
condition_for_fold(y_class_fold4)

bool_final=np.logical_and.reduce(bool_list)
#np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/bool_array_MoreThan5ActivesInactivesinEachFold_Chembl29.npy', bool_final)

#new_index for reduced file?
TargetID=1482 #TargetID in original file
index_list=np.array(list(range(0,3552)))
index_list_filtered=index_list[bool_final]
index=np.where(index_list_filtered==TargetID)
print(index)

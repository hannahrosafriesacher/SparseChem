#This file plots the positivity rate(=true positives) of different models.

from unittest import TestSuite
import sparsechem as sc
import numpy as np
import os
boolean=np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/SelectedTargets_5ActivesInactives/BooleanArray_TargetsWithMoreThan5ActivesInactivesInEachFold.npy').tolist()

print('load_file------------')
AUC_list=[]
#load class file
y_class=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
print(y_class.shape)

print('select_folds------------')
#select fold for class file
folding = np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')
keep    = np.isin(folding, 0)
y_class = sc.keep_row_data(y_class, keep) 

#load list of file name of small models
list_files_SM=os.listdir('/home/rosa/git/SparseChem/examples/chembl/predictions/models_LM_adam/repeats_bootstrap')
list_prediction_files=[]

#loading prediction files to list
for i in range(0, len(list_files_SM)):
    print(list_files_SM[i])
    current_file=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/predictions/models_LM_adam/repeats_bootstrap/'+str(list_files_SM[i]))[:, boolean]
    list_prediction_files.append(current_file.A)
    print(current_file.size)

print('average predictions_AUC-------------')
n=1
cumm_pred_now=np.zeros_like(list_prediction_files[0])
for k in range(len(list_prediction_files)):
    print('k:', k)
    print(list_prediction_files[k].size, list_prediction_files[k].shape)
    cumm_pred_now+=list_prediction_files[k]
    print(cumm_pred_now.size, cumm_pred_now.shape)
mean=cumm_pred_now/len(list_prediction_files)
np.save('/home/rosa/git/SparseChem/examples/chembl/predictions/models_LM_adam/repeats_bootstrap/sc_bootstrap_100models_h1000_ldo0.9_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0-class_TergetsWithMoreThan5InactivesActives.npy', mean)









'''
#obtaining Boolean file for Targets with more than 5 Actives and 5 Inactives in each fold

#load y_class file
y_class=sc.load_sparse('/home/aarany/git/chembl-pipeline/output/chembl_29/chembl_29_thresh.npy')
y_class=y_class.tocsc()
print(y_class.shape)

y_hat_LM=sc.load_sparse('predictions/h2000_ldo0.7_wdle-05_lr0.001_lrsteps10_ep20_fval1_fte0-class.npy')
y_hat_LM=y_hat_LM.tocsc()

y_hat_SingleSmallModel=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/predictions/models_logLikelihood/sc_run_h1000_ldo0.9_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy')
y_hat_SingleSmallModel=y_hat_SingleSmallModel.tocsc()

#seperate y_class by fold
folding=np.load('/home/aarany/git/chembl-pipeline/output/chembl_29/folding.npy')
fold=[0, 1, 2, 3, 4]
names_fold=[]
for i in fold:
    keep    = np.isin(folding, fold[i])
    globals()['y_class_fold%s' % i]= y_class[keep]
    names_fold.append(globals()['y_class_fold%s' % i])
print(y_class_fold0.shape, y_class_fold1.shape, y_class_fold2.shape,y_class_fold3.shape,y_class_fold4.shape)

#more than 5 actives/inactives per target?
#output: for all 5 folds: T/F-lists of length (#of targets) if they contain more than 5 actives and more than 5 inactives in this fold
for i in range(len(names_fold)):
    y_class_selected=names_fold[i].A.T
    globals()['fold%s' % i]=[]
    for k in range(y_class_selected.shape[0]):
        globals()['fold%s' % i].append(np.count_nonzero(y_class_selected[k,:]==1)>=5 and np.count_nonzero(y_class_selected[k,:]==-1)>=5)


'''
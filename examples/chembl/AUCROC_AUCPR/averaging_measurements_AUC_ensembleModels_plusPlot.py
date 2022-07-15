#This file compares the mean AUC_ROC values of a large model (hiddenSize=2000) and the average of 50 smaller models (hiddenSize=1000).
#We only consider those Targets with more than 5 Actives/Inactives

import sparsechem as sc
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import scipy.stats as sci
import matplotlib.pyplot as plt
import os

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

print('select_folds------------')
#load list of file name of small models
list_files_SM=os.listdir('/home/rosa/git/SparseChem/examples/chembl/predictions/models_SM_adam/repeats_bootstrap')

print('boolean------------')
#load boolean array
boolean=np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/SelectedTargets_5ActivesInactives/BooleanArray_TargetsWithMoreThan5ActivesInactivesInEachFold.npy').tolist()
y_class=y_class[:, boolean]

'''y_hat_one=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/predictions/models_logLikelihood/repeats/sc_repeat001_h1000_ldo0.9_wd1e-04_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy')[:, boolean].toarray()
y_hat_average_new=np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/models_logLikelihood/repeats_averages/average_sc_repeat001-050_h1000_ldo0.9_wd1e-0.4_lr0.001_lrsteps10_ep20_fva1_fte0-class_TargetsWithMoreThan5ActivesInactives_new.npy')
y_hat_average=np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/models_logLikelihood/repeats_averages/average_sc_repeat001-050_h1000_ldo0.9_wd1e-0.4_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy')[:, boolean]
print(y_hat_average_new.shape, y_hat_average.shape, y_hat_average.size, y_hat_average_new.size)

print(np.array_equal(y_hat_average, y_hat_average_new))
print(type(y_class), type(y_hat_one), type(y_hat_average))
print(y_class.shape, y_hat_one.shape, y_hat_average.shape)

roc_list_one=0
roc_list_av=0
for i in range(y_hat_one.shape[1]):
    y_hat_now=y_hat_one[:, i]
    y_hat_now_nonzero=y_hat_now[np.nonzero(y_hat_now)]
    y_class_now=y_class[:, i]
    y_class_now_nonzero=y_class_now[np.nonzero(y_class_now)]
    roc_list_one+=roc_auc_score(y_class_now_nonzero, y_hat_now_nonzero)
for k in range(y_hat_average.shape[1]):
    y_hat_now_av=y_hat_average[:, k]
    y_hat_now_nonzero_av=y_hat_now_av[np.nonzero(y_hat_now_av)]
    y_class_now_av=y_class[:, k]
    y_class_now_nonzero_av=y_class_now_av[np.nonzero(y_class_now_av)]
    roc_list_av+=roc_auc_score(y_class_now_nonzero_av, y_hat_now_nonzero_av)

print('average', roc_list_av/y_hat_average.shape[1])
print('one', roc_list_one/y_hat_one.shape[1])'''



print('list_predicition files------------')


list_prediction_files=[]
#loading prediction files to list
for i in range(len(list_files_SM)):
    current_file=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/predictions/models_SM_adam/repeats_bootstrap/'+str(list_files_SM[i]))[:, boolean]
    list_prediction_files.append(current_file.A)

print('average predictions_AUC-------------')
n=1
cumm_pred_now=np.zeros_like(list_prediction_files[0])
for k in range(len(list_prediction_files)):
    print('k:', k)
    #average predictions
    cumm_pred_now+=list_prediction_files[k]
    current_mean=cumm_pred_now/n
    #calculate AUC
    n+=1
    AUC_now=0
    for j in range(current_mean.shape[1]):
        y_hat_mean_currentTarget=current_mean[:,j]
        y_class_current_Target=y_class[:,j].A.flatten()
        y_hat_mean_nonzero=y_hat_mean_currentTarget[np.nonzero(y_class_current_Target)]
        y_class_nonzero=y_class_current_Target[np.nonzero(y_class_current_Target)]
        AUC_now+=roc_auc_score(y_class_nonzero, y_hat_mean_nonzero)
    AUC_list.append(AUC_now/current_mean.shape[1])
print(len(AUC_list))


lst = list(range(1,100+1))
plt.plot(lst, AUC_list)
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/models_SM_adam/AUC_ensembleLM_adam_bootstrap.png')

#mean=cumm_pred_now/len(list_prediction_files)
#print(mean.shape)

#np.save('/home/rosa/git/SparseChem/examples/chembl/predictions/models_logLikelihood/repeats_averages/average_sc_repeat001-050_h1000_ldo0.9_wd1e-0.4_lr0.001_lrsteps10_ep20_fva1_fte0-class_TargetsWithMoreThan5ActivesInactives_new.npy', mean)
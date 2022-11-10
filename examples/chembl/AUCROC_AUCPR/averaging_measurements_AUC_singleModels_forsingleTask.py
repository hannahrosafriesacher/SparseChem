#This file compares the mean AUC_ROC values of a large model (hiddenSize=2000) and the average of 50 smaller models (hiddenSize=1000).
#We only consider those Targets with more than 5 Actives/Inactives

import sparsechem as sc
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import scipy.stats as sci
import matplotlib.pyplot as plt
import os
from scipy import sparse

print('load_file------------')
AUC_list=[]
#load class file
y_class=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
print('y_class:', y_class.shape)
#y_hat=sparse.csr_matrix(np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/models_SM_adam/repeats_averages/average_sc_50repeats_h1000_ldo0.9_wd1e-0.4_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy'))
y_hat=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/predictions/models_SM_adam/sc_run_h1000_ldo0.9_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy')
print('y_hat:', y_hat.shape)
print('select_folds------------')
#select fold for class file
folding = np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')
keep    = np.isin(folding, 0)
y_class = sc.keep_row_data(y_class, keep) 
print(y_class.shape)

#print('boolean------------')
#load boolean array
#boolean=np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/SelectedTargets_5ActivesInactives/BooleanArray_TargetsWithMoreThan5ActivesInactivesInEachFold.npy').tolist()

#y_class_5IA=y_class[:, boolean]
y_class_5IA=y_class
y_hat_5IA=y_hat
print(y_class_5IA.shape, y_hat_5IA.shape)

AUC=0
PR=0
TargetID=1133

#y_hat_currentTarget=y_hat_5IA[:,TargetID].todense().A
y_hat_currentTarget=y_hat_5IA[:,TargetID].todense().A
y_class_current_Target=y_class_5IA[:,TargetID].todense().A
y_hat_nonzero=y_hat_currentTarget[np.nonzero(y_class_current_Target)]
#y_hat_nonzero=y_hat_currentTarget.todense()
print('1', y_hat_nonzero.shape)
y_class_nonzero=y_class_current_Target[np.nonzero(y_class_current_Target)]
print('2', y_class_nonzero.shape)

AUC+=roc_auc_score(y_class_nonzero, y_hat_nonzero)
precision, recall, thresholds = precision_recall_curve(y_class_nonzero, y_hat_nonzero)
PR+=auc(recall, precision)

AUC=AUC
PR=PR
print('AUC', AUC)
print('PR', PR)


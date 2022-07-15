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
boolean=np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/SelectedTargets_5ActivesInactives/BooleanArray_TargetsWithMoreThan5ActivesInactivesInEachFold.npy').tolist()
y_class=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')

targets=list(range(0, 888*4))
Targets_IDs=np.array(targets)[boolean]
print(Targets_IDs.shape)


print('select_folds------------')
#select fold for class file
folding = np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')
keep    = np.isin(folding, 0)
y_class = sc.keep_row_data(y_class, keep) 


print('boolean------------')
#load boolean array
y_class_5IA=y_class[:, boolean]


AUC=0
PR=0
i=0
for j in range(len(Targets_IDs)):
    y_hat_selected=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/predictions/plattScaling/LargeModel_adam_More5ActivesInactives/plattScaling_TargetID'+str(Targets_IDs[j]+1)+'.npy').todense()
    y_class_TargetID=y_class_5IA[:, j].todense()
    #y_hat_selected=y_hat_TargetID[np.nonzero(y_hat_TargetID)] 
    y_class_selected=y_class_TargetID[np.nonzero(y_class_TargetID)].T
    print(y_hat_selected.shape, y_class_selected.shape)
    
    i+=1
    AUC+=roc_auc_score(y_class_selected, y_hat_selected)
    precision, recall, thresholds = precision_recall_curve(y_class_selected, y_hat_selected)
    PR+=auc(recall, precision)

AUC=AUC/i
PR=PR/i
print('AUC', AUC)
print('PR', PR)



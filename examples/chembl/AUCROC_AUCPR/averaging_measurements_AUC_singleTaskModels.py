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

filename_y_class='/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/y_1482_reduced.npy'
filename_y_hat='/home/rosa/git/SparseChem/examples/chembl/predictions/model_SingleTask/SingleTask_1482_hiddenSizes_4_te_fold_0_lr_0.1_ep_50_batch_size200_class.npy'
filename_folding='/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/folding_1482_reduced.npy'
print('load_file------------')
#load class file
y_class=np.load(filename_y_class)
print(y_class.shape)
y_hat=np.load(filename_y_hat)

print('select_folds------------')
#select fold for class file
folding = np.load(filename_folding)
keep    = np.isin(folding, 0)
y_class = sc.keep_row_data(y_class, keep) 

AUC=roc_auc_score(y_class, y_hat)
precision, recall, thresholds = precision_recall_curve(y_class, y_hat)
PR=auc(recall, precision)
print('AUC', AUC)
print('PR', PR)


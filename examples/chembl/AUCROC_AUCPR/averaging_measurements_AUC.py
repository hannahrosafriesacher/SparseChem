#This file compares the mean AUC_ROC values of a large model (hiddenSize=2000) and the average of 50 smaller models (hiddenSize=1000).
#We only consider those Targets with more than 5 Actives/Inactives

import sparsechem as sc
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import scipy.stats as sci
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_class_TargetsWithMoreThan5ActivesInactivesinEachFold", "--y", "--y_classification", help=" Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_hat_average_SmallModels_TargetsWithMoreThan5ActivesInactivesinEachFold", "--y_hat_average", "--y_hat_average", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
parser.add_argument("--targetID", help="TargetID", type=int, required=True)
parser.add_argument("--y_hat_LargeModel_TargetsWithMoreThan5ActivesInactivesinEachFold", type=str)
parser.add_argument("--BooleanArray_TargetsWithMoreThan5ActivesInactivesinEachFold", type=str)
args = parser.parse_args()

#AUC_ROC values of smaller model ensemble
AUC_ROC_lists=[]
AUC_ROC_mean=[]

#load boolean array indicating which Targets have more than 5 Inactives/Actives in each fold
TargetList=np.load(args.BooleanArray_TargetsWithMoreThan5ActivesInactivesinEachFold)
print('TargetList:', TargetList.shape)

#load 'full files' (for AUC_ROc list: to get AUC_ROC values of specific targets via indexing) and files with already selected targets (for AUC_ROC mean)
y_class_full=sc.load_sparse('/home/aarany/git/chembl-pipeline/output/chembl_29/chembl_29_thresh.npy')
y_hat_average_full=sc.load_sparse('predictions/models_logLikelihood/repeats/average_sc_repeat001-050_h1000_ldo0.9_wd1e-0.4_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy')
y_hat_LM_full=sc.load_sparse('ROCAUC_comparison_large_small_models_-class.npy')
y_class_reduced = sc.load_sparse(args.y_class_TargetsWithMoreThan5ActivesInactivesinEachFold)
y_hat_average  = sc.load_sparse(args.y_hat_average_SmallModels_TargetsWithMoreThan5ActivesInactivesinEachFold)
y_hat_LM= sc.load_sparse(args.y_hat_LargeModel_TargetsWithMoreThan5ActivesInactivesinEachFold)

#select correct fold
folding = np.load(args.folding) if args.folding else None
keep    = np.isin(folding, 0)
y_class_reduced = sc.keep_row_data(y_class_reduced, keep) 
print('y_class_reduced', y_class_reduced.shape, 'y_hat_average',  y_hat_average.shape, 'y_hat_LM', y_hat_LM.shape)

#Sparse matrix of csc file
y_hat_average=y_hat_average.tocsc()
y_class_reduced=y_class_reduced.tocsc()
y_hat_LM=y_hat_LM.tocsc()
print('y_class_full', y_class_full.shape,'y_hat_average_full', y_hat_average_full.shape, ' y_hat_LM_full', y_hat_LM_full.shape)

#calculate AUC_ROC values for AUC_ROC_mean
for i in range(y_hat_average.shape[1]):
    y_hat_average_selected=y_hat_average[:,i].A
    y_class_selected=y_class_reduced[:,i].A
    y_hat_average_nonzero=y_hat_average_selected[np.nonzero(y_hat_average_selected)]
    y_class_nonzero=y_class_selected[np.nonzero(y_class_selected)]
    AUC_ROC_mean.append(roc_auc_score(y_class_nonzero, y_hat_average_nonzero))
#calculate AUC_ROC values for AUC_ROC_list (if Target has less than 5 actives/inactives per fold: append nan)
for j in range(y_hat_average_full.shape[1]):
    if TargetList[j]==True:
        y_hat_average_full_selected=y_hat_average_full[:, j].A
        y_class_full_selected=y_class_full[:, j].A
        y_class_full_selected = y_class_full_selected[keep, :]
        y_hat_average_full_nonzero=y_hat_average_full_selected[np.nonzero(y_hat_average_full_selected)]
        y_class_full_nonzero=y_class_full_selected[np.nonzero(y_class_full_selected)]
        AUC_ROC_lists.append(roc_auc_score(y_class_full_nonzero, y_hat_average_full_nonzero))
    else:
        AUC_ROC_lists.append('nan')

              
print('AUC_ROC_average_small_Models:', sum(AUC_ROC_mean)/len(AUC_ROC_mean))
print('AUC_ROC_average_small_Models: Number of tasks involved in averaging:', len(AUC_ROC_mean))
print('AUC_ROC_average_small_Models: overall number of tasks:', len(AUC_ROC_lists))
print('AUC_ROC specific Target:', AUC_ROC_lists[args.targetID])
print(len(AUC_ROC_lists))
print('---------------------------------------------------------------------------------')

#AUC_ROC values of larger model
AUC_ROC_lists_LM=[]
AUC_ROC_mean_LM=[]

#calculate AUC_ROC values for AUC_ROC_mean
for k in range(y_class_reduced.shape[1]):
    y_hat_LM_selected=y_hat_LM[:,k].A
    y_class_selected=y_class_reduced[:,k].A
    y_hat_LM_nonzero=y_hat_LM_selected[np.nonzero(y_hat_LM_selected)]
    y_class_nonzero=y_class_selected[np.nonzero(y_class_selected)]
    AUC_ROC_mean_LM.append(roc_auc_score(y_class_nonzero, y_hat_LM_nonzero))
#calculate AUC_ROC values for AUC_ROC_list (if Target has less than 5 actives/inactives per fold: append nan)
for m in range( y_hat_LM_full.shape[1]):
    if TargetList[m]==True:       
        y_hat_LM_full_selected=y_hat_LM_full[:, m].A
        y_class_full_selected=y_class_full[:, m].A
        y_class_full_selected = y_class_full_selected[keep, :]
        y_hat_LM_full_nonzero=y_hat_LM_full_selected[np.nonzero(y_hat_LM_full_selected)]
        y_class_full_nonzero=y_class_full_selected[np.nonzero(y_class_full_selected)]
        AUC_ROC_lists_LM.append(roc_auc_score(y_class_full_nonzero, y_hat_LM_full_nonzero))
    else:
        AUC_ROC_lists_LM.append('nan')
              
print('AUC_ROC_average_large_Model:', sum(AUC_ROC_mean_LM)/len(AUC_ROC_mean_LM))
print('AUC_ROC_average_large_Model: Number of tasks involved in averaging:', len(AUC_ROC_mean_LM))
print('AUC_ROC_average_large_Models: overall number of tasks:', len(AUC_ROC_lists_LM))
print(len(AUC_ROC_lists_LM))
print('AUC_ROC large Model specific Target:', AUC_ROC_lists_LM[args.targetID])

#Plot AUC_ROC_mean for small model ensemble in a histogram
#plt.hist(AUC_ROC_mean)
#plt.title('Ensemble of small models: \n AUC ROC of averaged predictions in test dataset')
#plt.xlabel('AUC ROC value of averaged predictions')
#plt.ylabel('absolute number')
#plt.savefig('predictions/models_logLikelihood/repeats_AUC_ROC_small models.png')
#This file obtaines calibrated predictions (Platt scaling)
import sparsechem as sc
import scipy.io
import scipy.special
import numpy as np
import pandas as pd
import torch
import sys
import argparse
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression as LR
from torch.utils.data import DataLoader
from scipy.special import expit

# argparse
parser = argparse.ArgumentParser(description="Using trained model to make predictions.")
parser.add_argument("--y_hat_test", help='Sparse pattern file with predictions for test fold.', type=str, required=True)
parser.add_argument("--y_hat_validation", help='Sparse pattern file with predictions for validation fold.', type=str, required=True)
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--TargetID", type=int, required=True)
parser.add_argument("--fold_va", help="Validation fold number", type=int, default=None)
args=parser.parse_args()

#load data
TargetID=args.TargetID
y_class = sc.load_sparse(args.y_class)
y_hat_va=sc.load_sparse(args.y_hat_validation)
y_hat_te=sc.load_sparse(args.y_hat_test)

#select rows in y_class that are included in validation fold
folding = np.load(args.folding) if args.folding else None
keep    = np.isin(folding, args.fold_va)
y_class_va = sc.keep_row_data(y_class, keep) 

#select Column by Target ID
y_class_va_TargetID=y_class_va[:, TargetID]
y_hat_va_TargetID=y_hat_va[:, TargetID]
y_hat_te_TargetID=y_hat_te[:, TargetID]

#keep rows with predictions
y_class_va_selected=y_class_va_TargetID[np.nonzero(y_hat_va_TargetID)]
y_hat_va_selected=y_hat_va_TargetID[np.nonzero(y_hat_va_TargetID)]
y_hat_te_selected=y_hat_te_TargetID[np.nonzero(y_hat_te_TargetID)]

#prepare data for logistic regression
y_class_va_selected=y_class_va_selected.A.flatten()
y_hat_va_selected=y_hat_va_selected.A.T
y_hat_te_selected=y_hat_te_selected.A.T

# inverse sigmoid of test and validation dataset
y_hat_va_withoutSigmoid=scipy.special.logit(y_hat_va_selected)
y_hat_te_withoutSigmoid=scipy.special.logit(y_hat_te_selected)

# train model
lr=LR()
lr.fit(y_hat_va_withoutSigmoid,y_class_va_selected)

#predict values with LM model for test dataset
y_hat_calibrated=lr.predict_proba(y_hat_te_withoutSigmoid)
y_hat_platt=y_hat_calibrated[:,np.nonzero(lr.classes_==1)].flatten()
y_hat_platt=np.reshape(y_hat_platt, (-1,y_hat_platt.shape[0]))

#obtain csr_matrix for platt predictiond
y_hat_platt=csr_matrix(y_hat_platt)

#save prediction as np-file in folder of y_hat_test
outprefix=args.y_hat_test[:-4]
np.save(str(outprefix)+'_plattScaling_TargetID'+str(TargetID), y_hat_platt)

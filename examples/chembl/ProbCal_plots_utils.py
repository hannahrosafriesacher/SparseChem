#This file 1)obtaines the counts and indices of the 100 tasks with the largest measurements, 2) obtaines the vector of one class of probability.

import sparsechem as sc
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_hat", type=str, required=True)
parser.add_argument("--TargetID", type=int, required=True)
parser.add_argument("--y_class", "--y", "--y_classification", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
args = parser.parse_args()

def split(arr, cond):
    return arr[cond]

#1) counts and indices of tasks with largest measurements:
#What are the Targets with the most compounds? Return first 100 targets:
TargetID=args.TargetID
y_class = sc.load_sparse(args.y_class)
y_hat=sc.load_sparse(args.y_hat)
y_hat=y_hat.tocsc()
counts=(y_hat.todense()>0).sum(0)

#What is y_hat/y_class of compounds in one specific class?
if args.folding is not None:
    folding = np.load(args.folding) if args.folding else None
    keep    = np.isin(folding, args.predict_fold)
    y_class = sc.keep_row_data(y_class, keep) 
sorted_counts_indices=np.argsort(counts*-1)
#print(sorted_counts_indices[:,0:100],sorted_counts_indices.shape)

#2)Vector of one class:
#How does the vector of a particular class look like? 
y_hat=y_hat.tocsc()
y_class=y_class.tocsc()

y_hat_TargetID=y_hat[:, TargetID]
y_class_TargetID=y_class[:, TargetID]
y_hat_selected=y_hat_TargetID[np.nonzero(y_hat_TargetID)] 
y_class_selected=y_class_TargetID[np.nonzero(y_hat_TargetID)]

#print(split(y_class_selected,np.logical_and(y_hat_selected>=0.9, y_hat_selected<1.0)).flatten())


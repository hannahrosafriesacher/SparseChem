import numpy as np
import sparsechem as sc
import matplotlib.pyplot as plt
import argparse
import math

parser = argparse.ArgumentParser(description="Obtaining Histograms for IC50 values of a target")
parser.add_argument("--targetID", help="TargetID", type=int, required=True)
parser.add_argument("--fold", type=int)
args = parser.parse_args()

y_regr=sc.load_sparse("chembl_29_regr.npy")

folding=np.load("/home/aarany/git/chembl-pipeline/output/chembl_29/folding.npy")
keep    = np.isin(folding, args.fold)
y_regr = sc.keep_row_data(y_regr, keep) 

y_regr_TargetId=y_regr[:, args.targetID]
y_regr_selected=y_regr_TargetId[np.nonzero(y_regr_TargetId)]
print(y_regr_selected.shape)
y_regr_selected=y_regr_selected.A.flatten()
print(np.arange(np.ceil(min(y_regr_selected)+0.01)-1, np.ceil(max(y_regr_selected)+1)))

fig, ax= plt.subplots()
binwidth=10
ax.hist(y_regr_selected, bins=np.arange(np.ceil(min(y_regr_selected)+0.01)-1, np.ceil(max(y_regr_selected))+1), rwidth=1)
ax.set_xlabel('pIC50')
ax.set_ylabel('absolute number')
ax.set_title("Number of bioactivities: " + str(y_regr_selected.shape[0]))
plt.savefig('./IC50_plots/IC50_Hist_'+str(args.targetID)+'_fold'+str(args.fold)+'.png')


import numpy as np
import sparsechem as sc
import matplotlib.pyplot as plt
import argparse

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
y_regr_selected=y_regr_selected.A.flatten()
y_regr_sort=np.sort(y_regr_selected)

X=np.array([list(range(1,y_regr_selected.shape[0]+1))]).T.flatten()
print(y_regr_sort)


fig, ax= plt.subplots()
ax.set_xlabel('Measurements in Validation fold')
ax.set_ylabel('pIC50')
ax.scatter(x=X, y=y_regr_sort)
ax.set_title("Number of bioactivities: " + str(y_regr_selected.shape[0]))
plt.savefig('./IC50_plots/IC50_DotPlots_'+str(args.targetID)+'_fold'+str(args.fold)+'.png')

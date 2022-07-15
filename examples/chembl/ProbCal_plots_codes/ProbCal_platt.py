#This file obtaines calibrated predictions (Platt scaling)
import sparsechem as sc
import scipy.io
import scipy.special
import numpy as np
import argparse
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression as LR
import os
# argparse
parser = argparse.ArgumentParser(description="Using trained model to make predictions.")
parser.add_argument("--y_hat_test", help='Sparse pattern file with predictions for test fold.', type=str, required=True)
parser.add_argument("--y_hat_validation", help='Sparse pattern file with predictions for validation fold.', type=str, required=True)
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--fold_va", help="Validation fold number", type=int, default=None)
parser.add_argument("--fold_te", help="Test fold number", type=int, default=None)
args=parser.parse_args()

targets=list(range(0, 888*4))
print(len(targets))

#select Test and Validation fold
y_class = sc.load_sparse(args.y_class)
y_hat_va=sc.load_sparse(args.y_hat_validation)
y_hat_te=sc.load_sparse(args.y_hat_test)
    
#select rows in y_class that are included in validation /test fold
folding = np.load(args.folding) if args.folding else None
keep_va   = np.isin(folding, args.fold_va)
keep_te = np.isin(folding, args.fold_te)
y_class_va = sc.keep_row_data(y_class, keep_va) 
y_class_te = sc.keep_row_data(y_class, keep_te)
print('1', y_hat_va.shape, y_class_va.shape, y_class_te.shape, y_hat_te.shape)

#keep targets with more than 5 actives/inactives
boolean=np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/SelectedTargets_5ActivesInactives/BooleanArray_TargetsWithMoreThan5ActivesInactivesInEachFold.npy').tolist()
print('boolean', len(boolean))

y_hat_va=y_hat_va[:, boolean]
y_hat_te=y_hat_te[:, boolean]
y_class_va=y_class_va[:, boolean]
y_class_te=y_class_te[:, boolean]
Targets_IDs=np.array(targets)[boolean]

print('5INactiAct', y_hat_va.shape, y_hat_te.shape, y_class_va.shape, y_class_te.shape, len(Targets_IDs), y_hat_va.shape[1])

#for each target: new y_hat file (platt calibrated)
for i in range(y_hat_va.shape[1]):
    print(i)
    #select Column by Target ID
    TargetID=targets[i]
    y_class_te_TargetID=y_class_te[:, TargetID] 
    y_class_va_TargetID=y_class_va[:, TargetID]
    y_hat_va_TargetID=y_hat_va[:, TargetID]
    y_hat_te_TargetID=y_hat_te[:, TargetID]
    print('2', y_class_te_TargetID.shape, y_class_va_TargetID.shape, y_hat_va_TargetID.shape, y_hat_te_TargetID.shape)

    #keep rows with predictions
    y_class_te_selected=y_class_te_TargetID[np.nonzero(y_class_te_TargetID)]
    y_class_va_selected=y_class_va_TargetID[np.nonzero(y_class_va_TargetID)]
    y_hat_va_selected=y_hat_va_TargetID[np.nonzero(y_hat_va_TargetID)]
    y_hat_te_selected=y_hat_te_TargetID[np.nonzero(y_hat_te_TargetID)]
    print('3', y_class_te_selected.shape, y_class_va_selected.shape, y_hat_va_selected.shape, y_hat_te_selected.shape)

    #Numbers of Bioactivities in Val Fold
    bioactivities=y_class_va_selected.shape
    numPos=y_class_va_selected[y_class_va_selected==1]
    numNeg=y_class_va_selected[y_class_va_selected==-1]
    print('Nr of Bioactivities in Val fold:', bioactivities, 'Num of Positives in Val Fold:', numPos.shape, 'Num of Negatives in val Fold:', numNeg.shape)

    #prepare data for logistic regression
    y_class_te_selected=y_class_te_selected.A.T
    y_class_va_selected=y_class_va_selected.A.T
    y_hat_va_selected=y_hat_va_selected.A.T
    y_hat_te_selected=y_hat_te_selected.A.T
    print('4', y_class_te_selected.shape, y_class_va_selected.shape, y_hat_va_selected.shape, y_hat_te_selected.shape)

    # inverse sigmoid of test and validation dataset
    y_hat_va_withoutSigmoid=scipy.special.logit(y_hat_va_selected)
    y_hat_te_withoutSigmoid=scipy.special.logit(y_hat_te_selected)
    print('5', y_hat_va_withoutSigmoid.shape, y_hat_te_withoutSigmoid.shape)
    print('6', y_hat_va_withoutSigmoid.shape,y_class_va_selected.shape)

    # train model
    lr=LR()
    lr.fit(y_hat_va_withoutSigmoid,y_class_va_selected)
    
    #predict values with LM model for test dataset
    y_hat_calibrated=lr.predict_proba(y_hat_te_withoutSigmoid)
    y_hat_platt=y_hat_calibrated[:,np.nonzero(lr.classes_==1)].flatten()
    y_hat_platt=np.reshape(y_hat_platt, (-1,y_hat_platt.shape[0]))

    #------------------------------------------------------------------------------
    #insert calibrated values in array of size y_hat (all compounds)

    print('Mean accuracy in Validation fold:', lr.score(y_hat_va_withoutSigmoid,y_class_va_selected))
    print('Mean accuracy in Test fold:', lr.score(y_hat_te_withoutSigmoid,y_class_te_selected))
    #obtain csr_matrix for platt predictiond
    y_hat_platt=csr_matrix(y_hat_platt.T)

    #save prediction as np-file in folder of y_hat_test
    outprefix=args.y_hat_test[:-4]
    np.save('/home/rosa/git/SparseChem/examples/chembl/predictions/plattScaling/LargeModel_More5ActivesInactives/'+'plattScaling_TargetID'+str(Targets_IDs[i]+1), y_hat_platt)


 


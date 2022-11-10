#This file obtain the ACE/ECE of a model for a given target.
import sparsechem as sc
import numpy as np
import argparse
import scipy.sparse

parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_hat", help="predicted Values", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
parser.add_argument("--targetID", help="TargetID", type=int, required=True)
args = parser.parse_args()

#load data (true values/ predictions)
TargetID=args.targetID
y_class = sc.load_sparse(args.y_class)
y_hat  = sc.load_sparse(args.y_hat)
#y_hat=scipy.sparse.csr_matrix(np.load(args.y_hat))
print(y_class.shape)
print(y_hat.shape)

#select correct fold for class dataset
folding = np.load(args.folding) if args.folding else None
keep    = np.isin(folding, args.predict_fold)
y_class = sc.keep_row_data(y_class, keep) 

#Sparse matrix of csc file
#y_hat_TargetID=y_hat.T.tocsc()
y_hat_TargetID=y_hat.tocsc()
y_class=y_class.tocsc()
print(y_hat_TargetID.shape, y_class.shape)


#specify Target and selecting nonzero values
#y_hat_TargetID=y_hat[:, TargetID]
y_class_TargetID=y_class[:, TargetID]
y_hat_TargetID=y_hat[:,TargetID]
print(y_class_TargetID.shape, y_hat_TargetID)

y_hat_selected=y_hat_TargetID[np.nonzero(y_class_TargetID)] 
y_class_selected=y_class_TargetID[np.nonzero(y_class_TargetID)]

print(y_hat_selected.shape, y_class_selected.shape)
#---------Some Useful Functions---------------
#split array according to condition
def split(arr, cond):
    return arr[cond]

#Calculate positive ratio (=accuracy)
#if there are no measurements (=no predictions) in a split: add 0 to acc list
#Note: if 0 is added to the list, the difference between acc and  conf is the conf of this split
def posRatio(arr, dimension):
    if np.unique(arr, axis=dimension).shape[dimension]>1:
        return (arr==1).sum()/arr.shape[dimension]
    else:
        return np.array(0)

#Calculate Mean of Probablities in Column (=confidence)
#if there are no measurements(=no predictions) in a split: the confidence is calculated from the values list
def ProbMean(arr, dimension, ind):
    if arr.shape[dimension]!=0:
        mean=np.mean(arr)
        return(mean)
    else:
        return values[ind]+0.5


#-----------------------ECE-----------------------
clas=[]
prob=[]
acc=[]
conf=[]
values=[0.0, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
j=0
k=0
i=0
#split values according to values-list (0.0, 0.1, 0.2...) 
for j in range(10):
    clas.extend(split(y_class_selected,np.logical_and(y_hat_selected>=values[j], y_hat_selected<values[j+1])).flatten())
    prob.extend(split(y_hat_selected,np.logical_and(y_hat_selected>=values[j], y_hat_selected<values[j+1])).flatten())
    j+=1
#Obtain positive ratio (=acc calculated from true values) and 
# probablity mean (=conf calculated from predictions) for each split
for k in range(10):
    acc.append(posRatio(clas[k], 1))
    conf.append(ProbMean(prob[k], 1, k))
    k+=1

#obtain ACE for this specific target:
ece=0
for i in range(10):
    ece+=(np.abs(np.array(acc[i])-np.array(conf[i]))*clas[i].shape[1])
    #      |               acc(b)-         conf(n)| * nb
ece=ece/y_class_selected.shape[1]
#   sumofECE/N
print('ECE of Target', TargetID, ece)


##-----------------------ACE-----------------------
y_hat_selected=y_hat_selected.A.flatten()
y_class_selected=y_class_selected.A.flatten()

#sort class and hat file by ascending probablity values in hat file
index_sort_y_hat=np.argsort(y_hat_selected)
y_hat_sorted=y_hat_selected[index_sort_y_hat]
y_class_sorted=y_class_selected[index_sort_y_hat]

#divide in 10 classes with equal numbers of predictions
y_hat_split=np.array_split(y_hat_sorted, 10)
y_class_split=np.array_split(y_class_sorted, 10)

acc_ace=[]
conf_acc=[]

#Obtain positive ratio (=acc calculated from true values) and 
#probablity mean (=conf calculated from predictions) for each split
for m in range(10):
    acc_ace.append(posRatio(y_class_split[m], 0))
    conf_acc.append(ProbMean(y_hat_split[m], 0, m))

acc_ace=np.array(acc_ace)
conf_acc=np.array(conf_acc)

#obtain ACE for this specific target:
ace=np.sum(np.abs(acc_ace-conf_acc))/10
#     SumOverAllR(|acc(b)-conf(b)|)/R
print('ACE of Target', TargetID, ace)

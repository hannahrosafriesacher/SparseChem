##This file obtain the mean ACE/ECE of a model.
import sparsechem as sc
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import sparse

parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_hat", help="predicted Values", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
args = parser.parse_args()

#load data (true values/ predictions)
boolean=np.load('/home/rosa/git/SparseChem/examples/chembl/predictions/SelectedTargets_5ActivesInactives/BooleanArray_TargetsWithMoreThan5ActivesInactivesInEachFold.npy').tolist()
y_class = sc.load_sparse(args.y_class)
y_hat = sc.load_sparse(args.y_hat)
#y_hat  = sparse.csr_matrix(np.load(args.y_hat))
values=[-0.1, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(y_class.shape, y_hat.shape)

#select correct fold for class dataset
folding = np.load(args.folding) if args.folding else None
keep    = np.isin(folding, args.predict_fold)
y_class = sc.keep_row_data(y_class, keep) 

#Sparse matrix of csc file
#y_hat=y_hat.tocsc()
y_class=y_class.tocsc()[:, boolean]
y_hat=y_hat.tocsc()[:, boolean]
#select Targets with more than 5 Actives/inactives
y_hat_5IA=y_hat
y_class_5IA=y_class
print(y_hat_5IA.shape, y_class_5IA.shape)


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


##-----------------------ECE-----------------------
#list of ece-values for each target
ECE_list=[]
#iterate through targets:
b=0
for b in range(y_hat_5IA.shape[1]):
    #specify Target and selecting nonzero values
    y_hat_TargetID=y_hat_5IA[:, b].todense()
    y_class_TargetID=y_class_5IA[:, b].todense()
    print(y_hat_TargetID.shape, y_class_TargetID.shape)
    y_hat_selected=y_hat_TargetID[np.nonzero(y_class_TargetID)]
    y_class_selected=y_class_TargetID[np.nonzero(y_class_TargetID)]

    
    clas=[]
    prob=[]
    acc=[]
    conf=[]
    j=0
    k=0
    i=0
    ece=0

    #split values according to values-list (0.0, 0.1, 0.2...) for current target
    for j in range(10):
        clas.extend(split(y_class_selected,np.logical_and(y_hat_selected>=values[j], y_hat_selected<values[j+1])).flatten())
        prob.extend(split(y_hat_selected,np.logical_and(y_hat_selected>=values[j], y_hat_selected<values[j+1])).flatten())
        j+=1

    #Obtain positive ratio (=acc calculated from true values) and 
    #probablity mean (=conf calculated from predictions) for each split for current target
    for k in range(10):
        acc.append(posRatio(clas[k], 1))
        conf.append(ProbMean(prob[k], 1, k))
        k+=1

    #obtain ECE for current target:
    for i in range(len(acc)):
    
        #the final ECE is divided by number of datapoints
        #if acc!=0:
        ece+=(np.abs(np.array(acc[i])-np.array(conf[i]))*clas[i].shape[1])
        #      |               acc(b)-         conf(n)| * nb
    #the final ECE is divided by number of datapoints
    ece=ece/y_class_selected.shape[1]
    #   sumofECE/N

    #Append ECE to list of ECEs (one ECE per target)#
    ECE_list.append(ece)

##-----------------------ACE-----------------------
ACE_list=[]
c=0
for c in range(y_hat_5IA.shape[1]):

    #specify Target and selecting nonzero values
    y_hat_TargetID=y_hat_5IA[:, c]
    y_class_TargetID=y_class_5IA[:, c]

    y_hat_selected=y_hat_TargetID[np.nonzero(y_hat_TargetID)] 
    y_class_selected=y_class_TargetID[np.nonzero(y_class_TargetID)]

    y_hat_selected=y_hat_selected.A.flatten()
    y_class_selected=y_class_selected.A.flatten()

    #sort class and hat file by ascending probablity values in hat file for current target:
    index_sort_y_hat=np.argsort(y_hat_selected)
    y_hat_sorted=y_hat_selected[index_sort_y_hat]
    y_class_sorted=y_class_selected[index_sort_y_hat]

    #divide in 10 classes with equal numbers of predictions for current target:
    y_hat_split=np.array_split(y_hat_sorted, 10)
    y_class_split=np.array_split(y_class_sorted, 10)

    acc_ace=[]
    conf_acc=[]

    #Obtain positive ratio (=acc calculated from true values) and 
    #probablity mean (=conf calculated from predictions) for each split for current target:
    for m in range(10):
        acc_ace.append(posRatio(y_class_split[m], 0))
        conf_acc.append(ProbMean(y_hat_split[m], 0,  m))

    acc_ace=np.array(acc_ace)
    conf_acc=np.array(conf_acc)

    #obtain ACE for current target:
    ace=np.sum(np.abs(acc_ace-conf_acc))/10
    #     SumOverAllR(|acc(b)-conf(b)|)/R

    #Append ECE to list of ACEs (one ACE per target)
    ACE_list.append(ace)
    c+=1


ECE_list=np.array(ECE_list)
ACE_list=np.array(ACE_list)

#calculate mean of ECE/ACE list
ECE_mean=np.mean(ECE_list)
ACE_mean=np.mean(ACE_list)

print('ECE_mean:', ECE_mean)
print('ACE_mean:', ACE_mean)

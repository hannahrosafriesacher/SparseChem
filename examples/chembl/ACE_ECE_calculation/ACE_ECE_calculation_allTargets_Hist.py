#This file obtain the ACE/ECE of a model for a given target.
import sparsechem as sc
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_hat_SM", help="predicted Values", type=str, default=None)
parser.add_argument("--y_hat_LM", help="predicted Values", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
args = parser.parse_args()

#load data (true values/ predictions)
y_class = sc.load_sparse(args.y_class)
y_hat_SM  = sc.load_sparse(args.y_hat_SM)
y_hat_LM= sc.load_sparse(args.y_hat_LM)

#select correct fold for class dataset
folding = np.load(args.folding) if args.folding else None
keep    = np.isin(folding, args.predict_fold)
y_class = sc.keep_row_data(y_class, keep) 

#Sparse matrix of csc file
y_hat_SM=y_hat_SM.tocsc()
y_hat_LM=y_hat_LM.tocsc()
y_class=y_class.tocsc()


ECE_list_SM=[]
ECE_list_LM=[]

ACE_list_SM=[]
ACE_list_LM=[]

y_hat_list=[y_hat_SM, y_hat_LM]
a=0
#-------------------------------------------------------------------------------------------------
for a in range(len(y_hat_list)):
    target_id=0
    for target_id in range(y_class.shape[1]):
        #specify Target and selecting nonzero values
        y_hat_TargetID=y_hat_list[a][:, target_id]
        y_class_TargetID=y_class[:, target_id]

        y_hat_selected=y_hat_TargetID[np.nonzero(y_hat_TargetID)] 
        y_class_selected=y_class_TargetID[np.nonzero(y_class_TargetID)]
        print('!!!!!!!!!!!', y_hat_selected.shape, y_class_selected.shape)
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

        if a == 0:
            ECE_list_SM.append(ece)

        if a == 1:
            ECE_list_LM.append(ece)


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

        if a == 0:
            ACE_list_SM.append(ace)
        
        if a == 1:
            ACE_list_LM.append(ace)

print(len(ECE_list_LM), len(ECE_list_SM), len(ACE_list_LM), len(ACE_list_SM))

ECE_SM_allTargets=np.array(ECE_list_SM)
ECE_LM_allTargets=np.array(ECE_list_LM)
ACE_SM_allTargets=np.array(ACE_list_SM)
ACE_LM_allTargets=np.array(ACE_list_LM)

#difference between ECEs/ACEs of LM and SM for alle targets
ECE_diffLMSM= ECE_LM_allTargets-ECE_SM_allTargets
ACE_diffLMSM=ACE_LM_allTargets-ACE_SM_allTargets

index_sorted=np.argsort(ACE_diffLMSM)
ECE_diffLMSM_sorted=ECE_diffLMSM[index_sorted]
ACE_diffLMSM_sorted=ACE_diffLMSM[index_sorted]
#plt.plot(ECE_diffLMSM, color='r')
#plt.plot(ACE_diffLMSM, color='b')
#plt.savefig('./ACE_ECE_calculation/TEST.png')

#Plot in Histogram
plt.hist(ECE_diffLMSM, alpha=0.5, label='ECE')
plt.hist(ACE_diffLMSM, alpha=0.5, label= 'ACE')
plt.xlabel("Error", size=10)
plt.ylabel("Target Count", size=10)
plt.title("Difference between ACE/ECE \n of the LM and the SM ensemble including all targets")
plt.legend(loc='upper right')
plt.savefig('./ACE_ECE_calculation/HistComparison_ACE_ECE_LMvsSM.png')

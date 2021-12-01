import sparsechem as sc
import numpy as np
import argparse
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import math

from datetime import datetime

date=datetime.now().strftime('%Y_%m_%d-%I:%M:%S_%p')


parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_hat", help="predicted Values", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
parser.add_argument("--conf", help="Model conf file (.json or .npy)", type=str, required=True)
parser.add_argument("--targetID", help="TargetID", type=int, required=True)

#load data
args = parser.parse_args()
conf = sc.load_results(args.conf, two_heads=True)["conf"]
TargetID=args.targetID


y_class = sc.load_sparse(args.y_class)
y_hat  = sc.load_sparse(args.y_hat)

#select correct fold
if args.folding is not None:
    folding = np.load(args.folding) if args.folding else None
    keep    = np.isin(folding, args.predict_fold)
    y_class = sc.keep_row_data(y_class, keep) 


#Sparse matrix of scs file
y_hat=y_hat.tocsc()
y_class=y_class.tocsc()

y_hat_TargetID=y_hat[:, TargetID]
y_class_TargetID=y_class[:, TargetID]
y_hat_selected=y_hat_TargetID[np.nonzero(y_hat_TargetID)] 
y_class_selected=y_class_TargetID[np.nonzero(y_hat_TargetID)]


#Obtain positive count for compounds with predicted values betwen 0.0-0.1, 0.1-0.2, ...
def split(arr, cond):
    return arr[cond]
def posRatio(arr):    
    return (arr==1).sum()/arr.shape[1]
#split into positives and negatives in each class in the _y_class file
def selectPosNeg(arr):
    pos=np.count_nonzero(arr==1)
    neg=np.count_nonzero(arr==-1)
    return pos, neg


Zero_y_class=split(y_class_selected, y_hat_selected<0.1)
Zero_acc=posRatio(Zero_y_class)
Zero_y_class_pos, Zero_y_class_neg=selectPosNeg(Zero_y_class)

One_y_class=split(y_class_selected,np.logical_and(y_hat_selected>=0.1, y_hat_selected<0.2))
One_acc=posRatio(One_y_class)
One_y_class_pos, One_y_class_neg=selectPosNeg(One_y_class)

Two_y_class=split(y_class_selected,np.logical_and(y_hat_selected>=0.2, y_hat_selected<0.3))
Two_acc=posRatio(Two_y_class)
Two_y_class_pos, Two_y_class_neg=selectPosNeg(Two_y_class)

Three_y_class=split(y_class_selected,np.logical_and(y_hat_selected>=0.3, y_hat_selected<0.4))
Three_acc=posRatio(Three_y_class)
Three_y_class_pos, Three_y_class_neg=selectPosNeg(Three_y_class)

Four_y_class=split(y_class_selected, np.logical_and(y_hat_selected>=0.4, y_hat_selected<0.5))
Four_acc=posRatio(Four_y_class)
Four_y_class_pos, Four_y_class_neg=selectPosNeg(Four_y_class)

Five_y_class=split(y_class_selected,np.logical_and(y_hat_selected>=0.5, y_hat_selected<0.6))
Five_acc=posRatio(Five_y_class)
Five_y_class_pos, Five_y_class_neg=selectPosNeg(Five_y_class)

Six_y_class=split(y_class_selected,np.logical_and(y_hat_selected>=0.6, y_hat_selected<0.7))
Six_acc=posRatio(Six_y_class)
Six_y_class_pos, Six_y_class_neg=selectPosNeg(Six_y_class)

Seven_y_class=split(y_class_selected,np.logical_and(y_hat_selected>=0.7, y_hat_selected<0.8))
Seven_acc=posRatio(Seven_y_class)
Seven_y_class_pos, Seven_y_class_neg=selectPosNeg(Seven_y_class)

Eight_y_class=split(y_class_selected,np.logical_and(y_hat_selected>=0.8, y_hat_selected<0.9))
Eight_acc=posRatio(Eight_y_class)
Eight_y_class_pos, Eight_y_class_neg=selectPosNeg(Eight_y_class)

Nine_y_class=split(y_class_selected,np.logical_and(y_hat_selected>=0.9, y_hat_selected<1.0))
Nine_acc=posRatio(Nine_y_class)
Nine_y_class_pos, Nine_y_class_neg=selectPosNeg(Nine_y_class)

# Confidence Intervals for Positive ratio
# iterate through classes and calculate Variance of the mean
NumPos=[Zero_y_class_pos, One_y_class_pos, Two_y_class_pos, Three_y_class_pos, Four_y_class_pos, Five_y_class_pos, Six_y_class_pos, Seven_y_class_pos, Eight_y_class_pos, Nine_y_class_pos]
NumNeg=[Zero_y_class_neg, One_y_class_neg, Two_y_class_neg, Three_y_class_neg, Four_y_class_neg, Five_y_class_neg, Six_y_class_neg, Seven_y_class_neg, Eight_y_class_neg, Nine_y_class_neg]
StDev=[]
i=0
for i in range(10):
    newVar=0
    newVar=(NumPos[i]*NumNeg[i])/((NumPos[i]+NumNeg[i])^2*(NumPos[i]+NumNeg[i]+1))
    StDev.append(math.sqrt(newVar)*0.01)
    i+=1
print('SD for each class:', StDev)


#Prepare for Plotting
X_axis_bar= ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5','0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0-9', '0.9-1.0'] 
values=[Zero_acc, One_acc, Two_acc, Three_acc, Four_acc, Five_acc, Six_acc, Seven_acc, Eight_acc, Nine_acc]

X_axis_hist=['0.0', '0.1','0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
print('Plotted positive counts:', values)


#Plot 
fig,axs=plt.subplots(2, 1, gridspec_kw={'height_ratios':[1, 1]}, figsize=(12,9))
axs[0].bar(X_axis_bar, values, yerr=StDev)
axs[0].tick_params(axis='x', rotation=40)
axs[0].set_box_aspect(1)
axs[0].set_title('Positive Ratio')
axs[0].set_xlabel('predicted activity')
axs[0].set_ylabel('positive ratio')
heights, bins= np.histogram(np.transpose(y_hat_selected))

axs[1].bar(bins[:-1], heights.astype(np.float32)/heights.sum(), width=bins[1]-bins[0])
axs[1].set_title('Counts')
axs[1].set_xlabel('predicted activity')
axs[1].set_ylabel('bioactivity count')

fig.suptitle('Target-ID:'+ str(TargetID) + '/ Total Number of Bioactivities:' + str(y_hat_selected.shape[1]))
fig.tight_layout()
plt.savefig('./histograms/Count and_PositiveRate_TargetID'+str(TargetID)+'_'+str(date)+'.pdf')


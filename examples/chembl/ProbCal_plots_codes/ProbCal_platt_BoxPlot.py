#This file plots the positivity rate (=true positives) and the frequency in each class in a boxplot and a histogram after platt scaling.

import sparsechem as sc
import numpy as np
import argparse
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as sci
import matplotlib.gridspec as gridspec
from datetime import datetime

date=datetime.now().strftime('%Y_%m_%d-%I:%M:%S_%p')
parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_hat_platt", help="predicted Values after platt scaling", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
parser.add_argument("--targetID", help="TargetID", type=int, required=True)
args = parser.parse_args()

#load data
TargetID=args.targetID
y_class = sc.load_sparse(args.y_class)
y_hat_platt=sc.load_sparse(args.y_hat_platt)

#select correct fold for y_class
folding = np.load(args.folding) if args.folding else None
keep    = np.isin(folding, args.predict_fold)
y_class = sc.keep_row_data(y_class, keep) 

#Sparse Matrix to csc-fil
y_hat_platt=y_hat_platt.A
y_hat_platt=y_hat_platt.flatten()
y_class=y_class.tocsc()

#Selection of TargetID; Keep rows with measurements
y_class_TargetID=y_class[:, TargetID]
y_class_selected=y_class_TargetID[np.nonzero(y_class_TargetID)]
y_class_selected=y_class_selected.A.flatten()

#Split array according to condition
def split(arr, cond):
    return arr[cond]
#Calculate positive ratio
def posRatio(arr):
    return (arr==1).sum()/arr.shape[0]
#split into positives and negatives 
def selectPosNeg(arr):
    pos=np.count_nonzero(arr==1)
    neg=np.count_nonzero(arr==-1)
    return pos, neg
#count number of positives/negatives
def selectPos(arr):
    pos=np.count_nonzero(arr==1)
    return pos
def selectNeg(arr):
    neg=np.count_nonzero(arr==-1)
    return neg

#obtain postive Ratio and negative/positive count for each class of predicted probablities
clas=[]
acc=[]
NumPos=[]
NumNeg=[]
values=[0.0, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
j=0
k=0

for j in range(10):
    clas.append(split(y_class_selected,np.logical_and(y_hat_platt>=values[j], y_hat_platt<values[j+1])))
    j+=1
    values[j]
for k in range(10):
    acc.append(posRatio(clas[k]))
    NumPos.append(selectPos(clas[k]))
    NumNeg.append(selectNeg(clas[k]))
    k+=1

print(clas)

# Confidence Intervals for Positive ratio
# iterate through classes and calculate Variance of the mean
Q1=[]
Q3=[]
Me=[]
Med=[]
i=0
WhisLo=[]
WhisHi=[]
stats_box=[]
for i in range(10):
    Stats={}
    q1=0
    q3=0
    whislo=0
    whishi=0
    me=0
    med=0
    q1,q3=sci.beta.interval(0.5, NumPos[i], NumNeg[i])
    whislo, whishi=sci.beta.interval(0.95, NumPos[i], NumNeg[i])
    me=sci.beta.mean(NumPos[i], NumNeg[i])
    med=sci.beta.median(NumPos[i], NumNeg[i])

    Stats['med']=med
    Stats['q1']=q1
    Stats['q3']=q3
    Stats['whislo']=whislo
    Stats['whishi']=whishi

    stats_box.append(Stats)
    i+=1

#Specify labels for X-axis
X_axis_bar= ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5','0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0-9', '0.9-1.0'] 
 
fig, axs=plt.subplots(2, 1, figsize=(9,9))

# Boxplot for each 'probability class'
axs[0].bxp(stats_box, showfliers=False, meanline=True)
axs[0].set_xticklabels(X_axis_bar)
axs[0].set_title('Positive Ratio', fontsize='x-large')
axs[0].set_xlabel('predicted activity')
axs[0].set_ylabel('positive ratio')
axs[0].axline([1,0.05], [10,0.95], color='r', linestyle='--')

#Plot number of compounds of each 'class'
heights, bins= np.histogram(np.transpose(y_hat_platt))
axs[1].bar(bins[:-1], heights/heights.sum(), align='center', tick_label=X_axis_bar, width=bins[1]-bins[0])
axs[1].set_title('Counts', fontsize='x-large')
axs[1].set_xlabel('predicted activity')
axs[1].set_ylabel('relative frequency')

fig.suptitle('Target-ID:'+ str(TargetID) + '\n Total Number of Bioactivities:' + str(y_hat_platt.shape[0]),fontsize='xx-large' )
fig.tight_layout()

#save figure
plt.savefig('./ProbCal_plots/BoxPlot_Count and_PositiveRate_TargetID_plattScaling'+str(TargetID)+'_'+str(date)+'.png')


#This file plots the positivity rate(=true positives) of different models.

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
parser.add_argument("--y_class", "--y", "--y_classification", help=" Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
parser.add_argument("--targetID", help="TargetID", type=int, required=True)
args = parser.parse_args()

#list of prediction files and fold used for prediction
y_hat_list=['predictions/differentFolds/h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy', 'predictions/differentFolds/h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte2-class.npy', 'predictions/differentFolds/h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte3-class.npy','predictions/differentFolds/h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte4-class.npy']
predict_fold_list=[0,2,3,4]

#load data
TargetID=args.targetID
y_class= sc.load_sparse(args.y_class)

#Split array according to condition
def split(arr, cond):
    return arr[cond]
#Caculate positive ratio
def posRatio(arr):
    print((arr==1).sum()/arr.shape[1])
    return (arr==1).sum()/arr.shape[1]

x_scatter=[1,2,3,4,5,6,7,8,9,10]
y_scatter=[]

#iterate through list of files and list of predict folds
for i in range(len(y_hat_list)):
    y_hat=sc.load_sparse(y_hat_list[i])
    folding = np.load(args.folding) if args.folding else None
    keep    = np.isin(folding, predict_fold_list[i])
    y_class_selected = sc.keep_row_data(y_class, keep) 

    #Sparse matrix of csc file
    y_hat_selected=y_hat.tocsc()
    y_class_selected=y_class_selected.tocsc()
    
    #filter for specified TargetID and nonzero values
    y_hat_TargetID=y_hat_selected[:, TargetID]
    y_class_TargetID=y_class_selected[:, TargetID]
    y_hat_selected_Target=y_hat_TargetID[np.nonzero(y_hat_TargetID)] 
    y_class_selected_Target=y_class_TargetID[np.nonzero(y_hat_TargetID)]
    
    #obtain positive ratios for each 'probability class' in each file in the file list
    clas=[]
    acc=[]
    values=[0.0, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    j=0
    k=0

    for j in range(10):
        clas.extend(split(y_class_selected_Target,np.logical_and(y_hat_selected_Target>=values[j], y_hat_selected_Target<values[j+1])).flatten())
        j+=1
    
    for k in range(10):
        acc.append(posRatio(clas[k]))
        k+=1

    y_scatter.append(acc)

#prepare Values of X-axis for plotting
X_axis_bar= ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5','0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0-9', '0.9-1.0']

#Scatter plot: plot positive ratio for each 'probability class' in each file in the file list
fig=plt.figure()
ax1=fig.add_subplot(111)

ax1.scatter(x_scatter, y_scatter[0], c='b', label='test fold :0', s= 6)
ax1.scatter(x_scatter, y_scatter[1], c='r', label='test fold: 2', s= 6)
ax1.scatter(x_scatter, y_scatter[2], c='y', label='test fold: 3', s= 6)
ax1.scatter(x_scatter, y_scatter[3], c='g', label='test fold: 4', s= 6)
ax1.set_xticks([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
ax1.set_xticklabels(X_axis_bar)
ax1.set_title('Positive Ratios for Different Folds', fontsize='x-large' )
ax1.set_xlabel('predicted activity')
ax1.set_ylabel('positive ratio')
ax1.legend()
fig.suptitle('Target-ID:'+ str(TargetID),fontsize='xx-large' )
fig.tight_layout()

#save figure
plt.savefig('./ProbCal_plots/Scatter_differentFolds_TargetID'+str(TargetID)+'_'+str(date)+'.pdf')


#This file plots the positivity rate(=true positives) of many models in a Violin PLot
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
parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
parser.add_argument("--targetID", help="TargetID", type=int, required=True)
args = parser.parse_args()

#create list of [001,002,...,050]
li=["{:03}".format(j) for j in range(101)]
li=li[1:]
print(li)
#create list of file names with predictions
y_hat_list=[]
for i in li:
    y_hat_list.append('predictions/repeats/sc_repeat'+str(i)+'_h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy')

#load data
TargetID=args.targetID
y_class= sc.load_sparse(args.y_class)

#select correct fold and filter according to TargetID, filter for nonzero values
folding = np.load(args.folding) if args.folding else None
keep    = np.isin(folding, args.predict_fold)
y_class_selected = sc.keep_row_data(y_class, keep) 
y_class_selected=y_class_selected.tocsc()
y_class_TargetID=y_class_selected[:, TargetID]
y_class_selected_Target=y_class_TargetID[np.nonzero(y_class_TargetID)]

#split array according to condition
def split(arr, cond):
    return arr[cond]
#calculate positive ratio
def posRatio(arr):
    return (arr==1).sum()/arr.shape[1]
#split into positives and negatives in each class in the _y_class file
def selectPosNeg(arr):
    pos=np.count_nonzero(arr==1)
    neg=np.count_nonzero(arr==-1)
    return pos, neg

#iterate through list of prediction file names and calculate for each 'probablity class' in each file the positive ratio and the number of positives/negatives

y_scatter=[]
y_hat_counts=[]
y_hat_all=[]
for i in range(len(y_hat_list)):
    #load and prepare data
    y_hat=sc.load_sparse(y_hat_list[i])
    #select correct fold and filter according to TargetID, filter for nonzero values
    y_hat_selected=y_hat.tocsc()
    y_hat_TargetID=y_hat_selected[:, TargetID]
    y_hat_selected_Target=y_hat_TargetID[np.nonzero(y_hat_TargetID)] 
    y_hat_all.append(y_hat_selected_Target.A.flatten().T)

    # obtain positive ratio and positive/negative count for each 'probability class'
    count=[]
    clas=[]
    acc=[]
    values=[0.0, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    j=0
    k=0

    for j in range(10):
        clas.extend(split(y_class_selected_Target,np.logical_and(y_hat_selected_Target>=values[j], y_hat_selected_Target<values[j+1])).flatten())
        count.append(split(y_class_selected_Target,np.logical_and(y_hat_selected_Target>=values[j], y_hat_selected_Target<values[j+1])).shape[1])
        j+=1
 
    for k in range(10):
        acc.append(posRatio(clas[k]))
        k+=1
    y_scatter.append(acc)
    y_hat_counts.append(count)

y_hat_all=np.asarray(y_hat_all)
y_hat_mean=np.mean(y_hat_all, axis=0)

#take the mean of all probabilities predicted by the different models for each compound
clas_means=[]
means=[]
for m in range(10):
    clas_means.append(split(y_class_selected_Target.T, np.logical_and(y_hat_mean>=values[m], y_hat_mean<values[m+1])))
    m+=1

for n in range(10):
    means.append(posRatio(clas_means[n].T))
    n+=1
    
#Specify X-Axis labels for plotting
X_axis_bar= ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5','0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0-9', '0.9-1.0']
y_scatter_plot=np.array(y_scatter).T.tolist()
y_hat_counts_plot=np.array(y_hat_counts)

fig, axs=plt.subplots(2, 1, figsize=(9,9))
#plot positive ratio as violin plots and the positive ratio of the mean of all models as points
axs[0].violinplot(y_scatter_plot)
axs[0].set_title('Positive Ratio', fontsize='x-large')
axs[0].set_xlabel('predicted activity')
axs[0].set_ylabel('positive ratio')
axs[0].set_xticks(ticks=[1,2,3,4,5,6,7,8,9,10])
axs[0].set_xticklabels(labels=X_axis_bar)
axs[0].axline([1, 0.05], [10,0.95], color='r', linestyle='--')
axs[0].scatter(x=[1,2,3,4,5,6,7,8,9,10], y=means, color='r')

#plot measurement counts in each 'probability class' obtained by all models as Boxplot
axs[1].boxplot(y_hat_counts_plot, labels= X_axis_bar)
axs[1].set_title('Counts', fontsize='x-large')
axs[1].set_xlabel('predicted activity')
axs[1].set_ylabel('absolute frequency')
fig.suptitle('Target-ID:'+ str(TargetID) + '\n Total number of Bioactivities:'+ str(y_class_selected_Target.shape[1]), fontsize='xx-large')
fig.tight_layout()

#save figure
plt.savefig('./ProbCal_plots/ViolinPlotAndModelMean_TargetID'+str(TargetID)+'_'+str(date)+'.png')


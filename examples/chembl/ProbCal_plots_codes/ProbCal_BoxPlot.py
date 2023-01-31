#This file plots the positivity rate (=true positives) and the frequency in each class in a boxplot and a histogram.
import sparsechem as sc
import numpy as np
import argparse
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as sci


parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
#parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
#parser.add_argument("--y_hat", help="predicted Values", type=str, default=None)
#parser.add_argument("--folding", help="Folds for rows of y, optional. Needed if only one fold should be predicted.", type=str, required=False)
#parser.add_argument("--predict_fold", help="One or more folds, integer(s). Needed if --folding is provided.", nargs="+", type=int, required=False)
#parser.add_argument("--targetID", help="TargetID", type=int, required=True)
parser.add_argument("--equalSizedBins", help='True: equal number fo molecules in each bin/ False: equally spaced bins', type=bool, default=False)
args = parser.parse_args()



y_hat_selected=np.matrix([[0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5]])
y_class_selected=np.matrix([[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1]])

'''#load data
TargetID=args.targetID
y_class = sc.load_sparse(args.y_class)
y_hat  = sc.load_sparse(args.y_hat)
#print('Number of Bioactivities in Fold', y_hat.shape, y_class.shape)

#select correct fold
folding = np.load(args.folding) if args.folding else None
keep    = np.isin(folding, args.predict_fold)
y_class = sc.keep_row_data(y_class, keep) 

#Sparse matrix of csc file
y_hat=y_hat.tocsc()
y_class=y_class.tocsc()
#print('Number of Bioactivities in Fold', y_hat.shape, y_class.shape)

#specify Target and selecting nonzero values
y_hat_TargetID=y_hat[:, TargetID]
y_class_TargetID=y_class[:, TargetID]
#print(y_hat_TargetID.shape, y_class_TargetID.shape)


y_hat_selected=y_hat_TargetID[np.nonzero(y_class_TargetID)] 
y_class_selected=y_class_TargetID[np.nonzero(y_class_TargetID)]
#print('Number of Bioactivities', y_class_selected.shape)'''

#split array according to condition
def split(arr, cond):
    return arr[cond]
#Calculate positive ratio
def posRatio(arr):    
    return (arr==1).sum()/arr.shape[0]
#split into positives and negatives in each class
def selectPosNeg(arr):
    pos=np.count_nonzero(arr==1)
    neg=np.count_nonzero(arr==-1)
    return pos, neg
def ProbMean(arr):
    x=np.array(arr)
    y=np.mean(x)
    return(y)
#count positives/negatives in each class
def selectPos(arr):
    pos=np.count_nonzero(arr==1)
    #print('pos', pos)
    return pos
def selectNeg(arr):
    neg=np.count_nonzero(arr==-1)
    #print('neg', neg)
    return neg

#Obtain positive ratio and positive/negative counts for compounds with predicted values betwen 0.0-0.1, 0.1-0.2, ...

acc=[]

mean_pred=[]
NumPos=[]
NumNeg=[]
values=[0.0, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ratios=[]
j=0
k=0

if args.equalSizedBins:
    #sort class and hat file by ascending probablity values in hat file
    index_sort_y_hat=np.argsort(y_hat_selected)
    y_hat_sorted=y_hat_selected.A.flatten()[index_sort_y_hat]
    y_class_sorted=y_class_selected.A.flatten()[index_sort_y_hat]

    #divide in 10 classes with equal numbers of predictions
    hat=np.array_split(y_hat_sorted.flatten(), 10)
    clas=np.array_split(y_class_sorted.flatten(), 10)
    
else:
    clas=[]
    hat= []
    for j in range(10):
        clas.extend(split(y_class_selected,np.logical_and(y_hat_selected>=values[j], y_hat_selected<values[j+1])).flatten())
        hat.append(split(y_hat_selected,np.logical_and(y_hat_selected>=values[j], y_hat_selected<values[j+1])).flatten())
        j+=1

for k in range(10):
    acc.append(posRatio(clas[k]))
    mean_pred.append(np.mean(hat[k]))
    NumPos.append(selectPos(clas[k]))
    NumNeg.append(selectNeg(clas[k]))
    
    k+=1


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
    ratios.append(NumPos[i]/(NumNeg[i]+NumPos[i]))
    q1,q3=sci.beta.interval(0.5, NumPos[i], NumNeg[i])
    whislo, whishi=sci.beta.interval(0.95, NumPos[i], NumNeg[i])
    me=sci.beta.mean(NumPos[i], NumNeg[i])
    med=sci.beta.median(NumPos[i], NumNeg[i])
    if NumNeg[i]==0 and NumPos[i]!=0:
        Stats['med']=med
        Stats['mean']=1
        Stats['q1']=q1
        Stats['q3']=q3
        Stats['whislo']=whislo
        Stats['whishi']=whishi

    elif NumPos[i]==0 and NumNeg[i]!=0:
        Stats['med']=med
        Stats['mean']=0
        Stats['q1']=q1
        Stats['q3']=q3
        Stats['whislo']=whislo
        Stats['whishi']=whishi
    else:        
        Stats['med']=med
        Stats['mean']=me
        Stats['q1']=q1
        Stats['q3']=q3
        Stats['whislo']=whislo
        Stats['whishi']=whishi
        
    stats_box.append(Stats)
    i+=1

#Prepare for Plotting
counts=[]
if args.equalSizedBins:
    for i in clas:
        counts.append(i.shape[0])
else:
    for i in clas:
        counts.append(i.shape[1])
        
         
if args.equalSizedBins:
    X_axis_bar=[]
    for n in range(len(hat)):
        clas_now=hat[n]
        if n ==0:
            first=float(0)
            last=hat[n][-1:]
            X_axis_bar.append(str(first)+'-'+str(last)[1:-1][:6])
        elif n==(len(hat)-1):
            first=hat[n][1]
            last=1.00
            X_axis_bar.append(str(first)[:6]+'-'+str(last))
        else:
            first=hat[n][1]
            last=hat[n][-1:]
            X_axis_bar.append(str(first)[:6]+'-'+str(last)[1:-1][:6])
    
else:
    X_axis_bar= ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5','0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0-9', '0.9-1.0']

meanpointprops = dict(markerfacecolor='black', markeredgecolor='black')
#D=Diamond, H=Hexagon, 
medianprops = dict(linestyle='--',  color='black')

if args.equalSizedBins:  
    fig, axs=plt.subplots()
    #plot a box plot for each 'probability class'
    axs.bxp(stats_box, showmeans=False, showfliers=False, meanprops=meanpointprops, medianprops= medianprops)
    axs.set_xticklabels(X_axis_bar, rotation = 25, size=12, ha='right')
    axs.plot([1,2,3,4,5,6,7,8,9,10], ratios, '*', c='black')
    #axs.set_title('Positive Ratio', fontsize='x-large')
    axs.set_xlabel('Predicted activity',  size=16)
    axs.set_ylabel('Positive ratio',  size=16)
    axs.set_ylim(-0.05,1.05)
    axs.stairs(mean_pred, [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5], fill=False, color='firebrick', linestyle='--')
else:
    #fig, axs=plt.subplots(2, 1, figsize=(9,9))
    fig, axs=plt.subplots()
    #plot a box plot for each 'probability class'
    #change to axs[0] if the second plot is also plotted
    axs.bxp(stats_box, showmeans=False, showfliers=False,medianprops= medianprops)
    axs.set_xticklabels(X_axis_bar, rotation = 25, size=12)
    axs.plot([1,2,3,4,5,6,7,8,9,10], ratios, '*', c='black')
    #axs.set_title('Positive Ratio', fontsize='x-large')
    axs.set_xlabel('Predicted activity', size=16)
    axs.set_ylabel('Positive ratio', size=16)
    axs.set_ylim(-0.05,1.05)
    axs.stairs(mean_pred, [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5], fill=False, color='firebrick', linestyle='--')
    '''   #PLot number of compounds for each 'probability class'
    heights, bins= np.histogram(np.transpose(y_hat_selected))
    axs[1].bar(x=[1,2,3,4,5,6,7,8,9,10], height=counts, align='center', tick_label=X_axis_bar,color='grey')
    axs[1].set_title('Counts', fontsize='x-large')
    axs[1].set_xlabel('Predicted activity')
    axs[1].set_ylabel('Relative frequency')'''


#fig.suptitle('Target-ID:'+ str(TargetID) + '\n Total Number of Bioactivities:' + str(y_hat_selected.shape[1]),fontsize='xx-large' )
fig.tight_layout()

#save figure
'''if args.equalSizedBins==True:
    plt.savefig('/home/rosa/git/SparseChem/examples/chembl/ProbCal_plots/BoxPlot_Count and_PositiveRate_TargetID'+str(TargetID)+'_equalSizedBins.png')    
    plt.savefig('/home/rosa/git/Images_Publications/ProbCalibr_Dec2023/ACE_Calibration Plot_TargetID_'+ str(TargetID) +'.pdf' , format='pdf', dpi=800) 
else:
    plt.savefig('/home/rosa/git/SparseChem/examples/chembl/ProbCal_plots/BoxPlot_Count and_PositiveRate_TargetID'+str(TargetID)+'_equalSpacedBins.png')
    plt.savefig('/home/rosa/git/Images_Publications/ProbCalibr_Dec2023/ECE_Calibration Plot_TargetID_'+ str(TargetID) +'.pdf' , format='pdf', dpi=800)'''

plt.savefig('/home/rosa/git/SparseChem/examples/chembl/ProbCal_plots/Test.png')

from itertools import count
import sparsechem as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import pandas as pd
#import seaborn as sns

'''y_hat_list=['/home/rosa/git/SparseChem/examples/chembl/predictions/models_VSM_adam/sc_run_h200_ldo0.8_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy', '/home/rosa/git/SparseChem/examples/chembl/predictions/models_SM_adam/sc_run_h1000_ldo0.9_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy', '/home/rosa/git/SparseChem/examples/chembl/predictions/models_LM_adam/h2000_ldo0.7_wdle-05_lr0.001_lrsteps10_ep20_fval1_fte0-class.npy']
#y_hat_list=['/home/rosa/git/SparseChem/examples/chembl/predictions/models_VSM_adam/sc_run_h200_ldo0.8_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0-class.npy']
ECE=[]
ACE=[]
for y in y_hat_list:
    print(y)
    #load data (true values/ predictions)
    y_class = sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
    y_hat = sc.load_sparse(y)
    #y_hat  = sparse.csr_matrix(np.load(args.y_hat))
    values=[-0.1, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    #select correct fold for class dataset
    folding = np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')
    keep    = np.isin(folding, 0)
    y_class = sc.keep_row_data(y_class, keep) 

    #Sparse matrix of csc file
    #y_hat=y_hat.tocsc()
    y_class=y_class.tocsc()
    y_hat=y_hat.tocsc()


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
    for b in range(y_hat.shape[1]):
        #specify Target and selecting nonzero values
        y_hat_TargetID=y_hat[:, b].todense()
        y_class_TargetID=y_class[:, b].todense()
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
    for c in range(y_hat.shape[1]):

        #specify Target and selecting nonzero values
        y_hat_TargetID=y_hat[:, c]
        y_class_TargetID=y_class[:, c]

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


    ECE.append(ECE_list)
    ACE.append(ACE_list)
    print(len(ECE))
    print(len(ACE))

ECE_np=np.array(ECE)
ACE_np=np.array(ACE)

df_ECE=pd.DataFrame(ECE_np.T, columns=['ECE_VSM', 'ECE_SM', 'ECE_LM'])
df_ACE=pd.DataFrame(ACE_np.T, columns=['ACE_VSM', 'ACE_SM', 'ACE_LM'])

res=df_ECE.join(df_ACE)

sizes = pd.read_csv('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/RankedTargetsNrActivesInactives.csv')
sizes = sizes.set_index('InternalID')
sizes_sorted=sizes.sort_index()
res = res.join(sizes_sorted)

res_json_large=sc.load_results('/home/rosa/git/SparseChem/examples/chembl/models/models_LM_adam/repeats/sc_repeat001_h2000_ldo0.7_wd1e-05_lr0.001_lrsteps10_ep20_fva1_fte0.json')
res_json_small=sc.load_results('/home/rosa/git/SparseChem/examples/chembl/models/models_SM_adam/sc_run_h1000_ldo0.9_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0.json')
res_json_very_small=sc.load_results('/home/rosa/git/SparseChem/examples/chembl/models/models_VSM_adam/sc_run_h200_ldo0.8_wd0.0001_lr0.001_lrsteps10_ep20_fva1_fte0.json')
df_large = res_json_large['validation']['classification']
df_small = res_json_small['validation']['classification']
df_very_small = res_json_very_small['validation']['classification']


df_small = df_small.join(res)
df_large = df_large.join(res)
df_very_small=df_very_small.join(res)

print(df_large)

df_small_filtered=df_small[(df_large.num_pos > 9) & (df_large.num_neg > 9)]
df_large_filtered=df_large[(df_large.num_pos > 9) & (df_large.num_neg > 9)]
df_very_small_filtered=df_very_small[(df_large.num_pos > 9) & (df_large.num_neg > 9)]
#df_large = df_large[(df_large.num_pos > 9) & (df_large.num_neg > 9)]
#df_small = df_small[(df_small.num_pos > 9) & (df_small.num_neg > 9)]
#TODO: loade ECE/ACE


bounds=[0,1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
ticks=['0-1000','1000-2000','2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000-7000', '7000-8000', '8000-9000', '9000-10000']
ECE_list=[]
ACE_list=[]

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

for i in range(len(bounds)-1):
    small=df_small_filtered.loc[(df_small_filtered.NrOfBioactivities>=bounds[i])&(df_small_filtered.NrOfBioactivities<bounds[i+1])]
    large=df_large_filtered.loc[(df_large_filtered.NrOfBioactivities>=bounds[i])&(df_large_filtered.NrOfBioactivities<bounds[i+1])]
    difference_ECE=large.ECE_LM-small.ECE_SM
    difference_ACE=large.ACE_LM-small.ACE_SM
    difference_ECE = difference_ECE[~np.isnan(difference_ECE)]
    ECE_list.append(difference_ECE.to_numpy())
    ACE_list.append(difference_ACE.to_numpy())



plt.figure(figsize=(10, 5))
plt.axhline(y = 0, color = 'black', alpha=0.4, linestyle = '--')
bp1=plt.boxplot(ECE_list, positions=np.array([1,4,7,10,13,16,19,22,25,28]))
set_box_color(bp1,'#D7191C')
bp2=plt.boxplot(ACE_list, positions=np.array([2,5,8,11,14,17,20,23,26,29]))
set_box_color(bp2, '#2C7BB6')
    

plt.plot([], c='#D7191C', label='ECE')
plt.plot([], c='#2C7BB6', label='ACE')
plt.legend()

plt.xticks([1.5,4.5,7.5,10.5,13.5,16.5,19.5,22.5,25.5,28.5], ticks, rotation=45)
plt.xlim(-1, len(ticks)*3+1)
plt.xlabel('Number of Measurements')
plt.ylabel('Difference between CEs')

plt.tight_layout()
#plot things'''


'''plt.figure()
plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ECE_SM-df_small_filtered.ECE_VSM, s=4, c='#D7191C')
plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ACE_SM-df_small_filtered.ACE_VSM, s=4, c='#2C7BB6')
plt.axhline(y = 0, color = 'black', alpha=0.4, linestyle = '--')
plt.plot([], c='#D7191C', label='ECE')
plt.plot([], c='#2C7BB6', label='ACE')
plt.legend()
plt.xlabel('Number of Target-Specific Measurements')
plt.ylabel('Difference between CEs')
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/ECE_ACE_SM-VSM.png')


plt.cla()'''

'''plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ECE_LM-df_small_filtered.ECE_VSM, s=5)
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/ECE_LM-VSM.png')
plt.cla()

plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ACE_LM-df_small_filtered.ACE_SM, s=5)
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/ACE_LM-SM.png')
plt.cla()

plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ECE_LM-df_small_filtered.ECE_SM, s=5)
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/ECE_LM-SM.png')
plt.cla()

plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ECE_LM, s=5)
plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ACE_LM, s=5)
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/LM_ECEvsACE.png')
plt.cla()

plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ECE_SM, s=5)
plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ACE_SM, s=5)
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/SM_ECEvsACE.png')
plt.cla()

plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ECE_VSM, s=5)
plt.scatter(df_small_filtered.NrOfBioactivities, df_small_filtered.ACE_VSM, s=5)
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/VSM_ECEvsACE.png')
plt.cla()

#########################################################################################################################################################
res=pd.read_csv('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/ECE_ECE_ROC_AUC_SingleTask_vs_MultiTask', sep=';', index_col=0)
res=res.T
res=res.rename(columns={'Nr. Cmps':'NrCpds'})

Targets_large=res.loc[res.NrCpds>6000]
Targets_medium=res.loc[(res.NrCpds<6000)&(res.NrCpds>1500)]
Targets_small=res.loc[res.NrCpds<1500]

ACE_Tlarge=Targets_large.MT_SM_ACE-Targets_large.ST_ACE
ACE_Tmedium=Targets_medium.MT_SM_ACE-Targets_medium.ST_ACE
ACE_Tsmall=Targets_small.MT_SM_ACE-Targets_small.ST_ACE

ECE_Tlarge=Targets_large.MT_SM_ECE-Targets_large.ST_ECE
ECE_Tmedium=Targets_medium.MT_SM_ECE-Targets_medium.ST_ECE
ECE_Tsmall=Targets_small.MT_SM_ECE-Targets_small.ST_ECE


ECE=[ECE_Tlarge.to_numpy(), ECE_Tmedium.to_numpy(), ECE_Tsmall.to_numpy()]
ACE=[ACE_Tlarge.to_numpy(), ACE_Tmedium.to_numpy(), ACE_Tsmall.to_numpy()]


ticks = ['large Targets', 'medium Targets', 'small Targets']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()
plt.axhline(y = 0, color = 'black', alpha=0.4, linestyle = '--')

bpl = plt.boxplot(ECE, positions=((np.array(range(len(ECE))))*2.0-0.4), widths=0.6)
bpr = plt.boxplot(ACE, positions=((np.array(range(len(ECE))))*2.0+0.4), widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='ECE')
plt.plot([], c='#2C7BB6', label='ACE')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-1, len(ticks)*2)
plt.ylabel('Difference between CEs')
plt.tight_layout()

plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/Test.png')'''

#################################################################################################################################################
# ECCB Poster comparison ACEvsECe
#load data (true values/ predictions)
TargetID=1482
y_class = sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/chembl_29_thresh.npy')
y_hat  = sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/predictions/models_LM_adam/h2000_ldo0.7_wdle-05_lr0.001_lrsteps10_ep20_fval1_fte0-class.npy')
#y_hat=scipy.sparse.csr_matrix(np.load(args.y_hat))
print(y_class.shape)
print(y_hat.shape)

#select correct fold for class dataset
folding = np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/folding.npy')
keep    = np.isin(folding, 0)
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

ticks=['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
plt.figure(figsize=[6.4,5.5])
plt.cla()
plt.bar([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5], acc, color='#B6B3B3', width=1)
plt.stairs(edges=[0,1,2,3,4,5,6,7,8,9, 10], values=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85, 0.95], color='black',  linewidth=2)
plt.xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5], ticks, rotation=45)
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Ratio of Positives (grey)')
plt.tight_layout()
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/ECE.png', dpi=1000)



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

#original:ticks=['0.0-0.00043', '0.00043-0.00133', '0.00133-0.0036', '0.0036-0.0082',  '0.0082-0.0249', '0.0259-0.0945', '0.0945-0.35', '0.35-0.8', '0.8-0.93', '0.93-1.0']
#original:pos_ticks=[0+(conf_acc[0]/2),np.sum(conf_acc[:1])+(conf_acc[1]/2), np.sum(conf_acc[:2])+(conf_acc[2]/2), np.sum(conf_acc[:3])+(conf_acc[3]/2), np.sum(conf_acc[:4])+(conf_acc[4]/2), np.sum(conf_acc[:5])+(conf_acc[5]/2), np.sum(conf_acc[:6])+(conf_acc[6]/2), np.sum(conf_acc[:7])+(conf_acc[7]/2), np.sum(conf_acc[:8])+(conf_acc[8]/2),np.sum(conf_acc[:9])+(conf_acc[9]/2)]

#statt '...'
plt.figure(figsize=[6.4,5.5])
pos=[0,np.sum(conf_acc[:1]), np.sum(conf_acc[:2]), np.sum(conf_acc[:3]), np.sum(conf_acc[:4]), np.sum(conf_acc[:5]), np.sum(conf_acc[:6]), np.sum(conf_acc[:7]), np.sum(conf_acc[:8]),np.sum(conf_acc[:9])]
pos_stairs=[0,np.sum(conf_acc[:1]), np.sum(conf_acc[:2]), np.sum(conf_acc[:3]), np.sum(conf_acc[:4]), np.sum(conf_acc[:5]), np.sum(conf_acc[:6]), np.sum(conf_acc[:7]), np.sum(conf_acc[:8]),np.sum(conf_acc[:9]),np.sum(conf_acc)]
ticks=['0.0-0.0004', ' ', ' ', ' ', ' ', ' ', '...', '0.35-0.8', '0.8-0.93', '0.93-1.0']
pos_ticks=[0, np.sum(conf_acc[:1])+(conf_acc[1]/2), np.sum(conf_acc[:2])+(conf_acc[2]/2), np.sum(conf_acc[:3])+(conf_acc[3]/2), np.sum(conf_acc[:4])+(conf_acc[4]/2), np.sum(conf_acc[:5])+(conf_acc[5]/2), np.sum(conf_acc[:6])+(conf_acc[6]/2), np.sum(conf_acc[:7])+(conf_acc[7]/2), np.sum(conf_acc[:8])+(conf_acc[8]/2),np.sum(conf_acc[:9])+(conf_acc[9]/2)]
plt.bar(x=pos, height=acc_ace, width=conf_acc, align='edge', color='#B6B3B3')
plt.stairs(edges=pos_stairs, values=conf_acc, color='black',  linewidth=2)
plt.xticks(pos_ticks, ticks, rotation=45)
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Ratio of Positives (grey)')
plt.tight_layout()
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/Data_Comparison_SingleTask_MultiTask/ACE.png', dpi=1000)
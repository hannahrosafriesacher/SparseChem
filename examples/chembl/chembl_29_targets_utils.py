import pandas as pd
import numpy as np
import sparsechem as sc
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Obtaining Histograms for Probability Calibration for singular Taget")
parser.add_argument("--y_class", "--y", "--y_classification", help="Sparse pattern file for classification, optional. If provided returns predictions for given locations only (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--chemblID_List", type=str, default=None, required=False)
parser.add_argument("--targetID", help="TargetID", type=int, required=False)
parser.add_argument("--val_fold", required=False)
parser.add_argument("--test_fold", required=False)
parser.add_argument("--folding", type=str, required=False)
args = parser.parse_args()

#count and vizualize actives/inactives er target in datafile
'''number_inactives=[]
number_actives=[]
total_nonzero=[]

y_class=sc.load_sparse(args.y_class)
y_class=y_class.todense()
for i in range(888):
    print('-----------', i)
    #group1
    target_now_1=y_class[:,i]
    targets_now_nonzero_1=target_now_1[np.nonzero(target_now_1)]
    counts_nonzero_now_1=targets_now_nonzero_1.shape[1]
    counts_postive_now_1=np.nonzero(targets_now_nonzero_1==1)[1].shape[0]
    counts_negative_now_1=np.nonzero(targets_now_nonzero_1==-1)[1].shape[0]

    #group2
    target_now_2=y_class[:,(i+888)]
    targets_now_nonzero_2=target_now_2[np.nonzero(target_now_2)]
    counts_nonzero_now_2=targets_now_nonzero_2.shape[1]
    counts_postive_now_2=np.nonzero(targets_now_nonzero_2==1)[1].shape[0]
    counts_negative_now_2=np.nonzero(targets_now_nonzero_2==-1)[1].shape[0]

    #group3
    target_now_3=y_class[:,(i+1776)]
    targets_now_nonzero_3=target_now_3[np.nonzero(target_now_3)]
    counts_nonzero_now_3=targets_now_nonzero_3.shape[1]
    counts_postive_now_3=np.nonzero(targets_now_nonzero_3==1)[1].shape[0]
    counts_negative_now_3=np.nonzero(targets_now_nonzero_3==-1)[1].shape[0]

    #group4
    target_now_4=y_class[:,(i+2664)]
    targets_now_nonzero_4=target_now_4[np.nonzero(target_now_4)]
    counts_nonzero_now_4=targets_now_nonzero_4.shape[1]
    counts_postive_now_4=np.nonzero(targets_now_nonzero_4==1)[1].shape[0]
    counts_negative_now_4=np.nonzero(targets_now_nonzero_4==-1)[1].shape[0]

    total_nonzero.append([counts_nonzero_now_1, counts_nonzero_now_2, counts_nonzero_now_3, counts_nonzero_now_4])
    number_actives.append([counts_postive_now_1, counts_postive_now_2, counts_postive_now_3, counts_postive_now_4])
    number_inactives.append([counts_negative_now_1, counts_negative_now_2, counts_negative_now_3, counts_negative_now_4])

total_nonzero_np=np.array(total_nonzero)
number_actives_np=np.array(number_actives)
number_inactives_np=np.array(number_inactives)
np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Plots_Data_Distrubution/Large_chembl_29_Uniprot_countsTotal_per_target.npy', total_nonzero_np)
np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Plots_Data_Distrubution/Large_chembl_29_Uniprot_countsActives_per_target.npy', number_actives_np)
np.save('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Plots_Data_Distrubution/Large_chembl_29_Uniprot_countsInctives_per_target.npy', number_inactives_np)'''

total_nonzero_np=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Plots_Data_Distrubution/Large_chembl_29_Uniprot_countsTotal_per_target.npy')
number_actives_np=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Plots_Data_Distrubution/Large_chembl_29_Uniprot_countsActives_per_target.npy')
number_inactives_np= np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Plots_Data_Distrubution/Large_chembl_29_Uniprot_countsInctives_per_target.npy')
ratio=np.divide(number_actives_np,total_nonzero_np)

'''counts=0
for i in range(888):
    now_target=ratio[i, :]
    print(now_target)
    if np.any(now_target<0.1):
        if np.all(now_target>0.0):
            counts+=1
            print('TRUE')

print(counts)'''

ratio_flat=ratio.flatten()
ratio_threshold01=np.logical_and(ratio<0.01, ratio>0.0).flatten()
print(ratio_threshold01)
total_counts_flat=total_nonzero_np.flatten()
total_counts_threshold01=total_counts_flat[ratio_threshold01]
print(total_counts_threshold01.shape, ratio_threshold01)

fig, axs=plt.subplots()
axs.hist(total_counts_threshold01, bins=500)
axs.set_title('Bioactivity counts for targets with an active ratio < 0.01 %', fontsize='x-large')
axs.set_xlabel('Bioactivity counts')
axs.set_ylabel('Number of Targets')
#axs.set_xlim(left=0, right=10000)
#axs.set_xticks([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Plots_Data_Distrubution/Hist.png')



'''ar=list(range(1,889))
ratio_1=ratio[:, 0]
ratio_2=ratio[:, 1]
ratio_3=ratio[:, 2]
ratio_4=ratio[:, 3]
total_1=total_nonzero_np[:, 0]
total_2=total_nonzero_np[:, 1]
total_3=total_nonzero_np[:, 2]
total_4=total_nonzero_np[:, 3]


#scatter and bar plots: actives ratio anf bioactivity countd for each target
fig, axs=plt.subplots(2, 1, figsize=(9,9))

axs[0].scatter(x=ar, y=ratio_1, c='b', s=3, alpha=0.5, label='5.5 < pIC50 < 6.5')
axs[0].scatter(x=ar, y=ratio_2, c='g', s=3, alpha=0.5, label='6.5 < pIC50 < 7.5')
axs[0].scatter(x=ar, y=ratio_3, c='r', s=3, alpha=0.5, label='7.5 < pIC50 < 8.5')
axs[0].scatter(x=ar, y=ratio_4, c='k', s=3, alpha=0.5, label='8.5')
axs[0].set_title('Ratio of Actives for each target', fontsize='x-large')
axs[0].set_xlabel('Target IDs (internal)')
axs[0].set_ylabel('Ratio of actives')
axs[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
axs[0].legend()

print(total_1.shape)
axs[1].bar(x=ar, height=total_1, color='b', alpha=0.5,label='5.5 < pIC50 < 6.5')
axs[1].bar(x=ar, height=total_2, color='g', alpha=0.5,label='6.5 < pIC50 < 7.5')
axs[1].bar(x=ar, height=total_3, color='r', alpha=0.5,label='7.5 < pIC50 < 8.5')
axs[1].bar(x=ar, height=total_4, color='k', alpha=0.5,label='8.5')
axs[1].set_title('Bioactivity counts for each target', fontsize='x-large')
axs[1].set_xlabel('Target IDs (internal)')
axs[1].set_ylabel('Total numberof bioactives')
#axs[1].set_yticks([5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000])
#axs[1].set_ylim(top=70000)
axs[1].legend()
fig.tight_layout()
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Plots_Data_Distrubution/Ratio.png')
'''


'''y_class=sc.load_sparse(args.y_class)
print(y_class.shape)
counts_total_per_target=[]
counts_active_per_target=[]
counts_inactive_per_target=[]
for i in range(y_class.shape[1]):
    print('--------------------i', i)
    target=y_class[:, i]
    nonzero=target[np.nonzero(target)]
    print(nonzero.shape)
    counts_total=nonzero.shape[1]
    counts_total_per_target.append(counts_total)
    counts_active=np.count_nonzero(nonzero==1)
    print(counts_active)
    counts_active_per_target.append(counts_active)
    counts_inactive=np.count_nonzero(nonzero==-1)
    print(counts_inactive)
    counts_inactive_per_target.append(counts_inactive)
    
indeces_total=np.argsort(counts_total_per_target)
indeces_active=np.argsort(counts_active_per_target)
indeces_inactive=np.argsort(counts_inactive_per_target)
np.save('/home/rosa/git/SparseChem/examples/chembl/auxiliary_datafiles/indicesMaxValues_Chemble_Uniprot_total.npy', indeces_total)
np.save('/home/rosa/git/SparseChem/examples/chembl/auxiliary_datafiles/indicesMaxValues_Chemble_Uniprot_actives.npy', indeces_active)
np.save('/home/rosa/git/SparseChem/examples/chembl/auxiliary_datafiles/indicesMaxValues_Chemble_Uniprot_inactives.npy', indeces_inactive)'''

#val_fold=args.val_fold
#test_fold=args.test_fold
#folding=np.load(args.folding)
#keep_val=np.isin(folding, val_fold)
#keep_test=np.isin(folding, test_fold)
#y_class_val=sc.keep_row_data(y_class, keep_val)
#y_class_test=sc.keep_row_data(y_class, keep_test)


#targets_internalID=args.targetID
#targets_chemblID=pd.read_csv(args.chemblID_List, sep=',', header=None)
#targets_chemblID=targets_chemblID.values.tolist()
#targets_np_sparse=sc.load_sparse(args.y_class)
#targets_np=targets_np_sparse.tocsc()

#indices=list(range(0, len(targets_chemblID)))
#indices_forYclass=[]

#for i in indices:
#    for j in range(4):
#        indices_forYclass.append(indices[i])

'''#Which ChemblID corresponds to the  internal target ID?
ChemblID_index=indices_forYclass[targets_internalID]
ChemblID=targets_chemblID[ChemblID_index]
print('IntenalID_yclass:', targets_internalID)
print('IntenalID_yReg:', ChemblID_index)
print('ChemblID:', ChemblID[0])'''

#------------------------------------
'''#How many bioactivities?
targets_selected=targets_np_sparse[:, targets_internalID]
bioactivities=np.nonzero(targets_selected)[0]
print('Bioactivity count:', bioactivities.shape[0])

#How many positives?
positives=targets_selected[targets_selected==1]
print('Actives:', positives.shape[1])

#How many negatives?
negatives=targets_selected[targets_selected==-1]
print('Inactives:', negatives.shape[1])'''

#--------------------------------------
'''folding=np.load(args.folding)
keep_val=np.isin(args.folding,1)
y_class_val=sc.keep_row_data(targets_np_sparse, keep_val)
print(y_class_val)

#How many bioactivities in ValFold?
targets_selected_val=y_class_val[:, targets_internalID]
bioactivities_val=np.nonzero(targets_selected_val)[0]
print('Bioactivity count in val fold:', bioactivities.shape[0])
print(targets_selected_val)

#How many positives in ValFold?
positives_val=targets_selected_val[targets_selected_val==1]
print('Actives in val Fold:', positives_val.shape[1])


#How many negatives in ValFold?
negatives_val=targets_selected_val[targets_selected_val==-1]
print('Inactives in val Fold:', negatives_val.shape[1])'''

#-----------------------------------------------
'''folding=np.load(args.folding) if args.folding else None
keep_te=np.isin(folding, args.test_fold)
targets_selected_te=sc.keep_row_data(targets_selected, keep_te)
print(targets_selected_te)


#How many bioactivities in TestFold?
bioactivities_te=np.nonzero(targets_selected_te)[0]
print('Bioactivity count in test Fold:', bioactivities_te.shape[0])

#How many positives in TestFold?
positives_te=targets_selected_te[targets_selected_te==1]
print('Actives in test Fold:', positives_te.shape[1])


#How many negatives in TestFold?
negatives_te=targets_selected_te[targets_selected_te==-1]
print('Inactives in test Fold:', negatives_te.shape[1])'''






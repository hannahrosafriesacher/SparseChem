import numpy as np
import sparsechem as sc
import matplotlib.pyplot as plt

y_class_chembl=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Large_chembl_29_Uniprot_inactives_thresh_csr.npy')
print(y_class_chembl.shape)
ratios=[]
for i in range(0, y_class_chembl.shape[1]):
    print(i)
    y_class_TargetID=y_class_chembl[:, i]
    if y_class_TargetID.size==0:
        ratios.append(1)
    else:
        y_class_TargetID_nonzero=y_class_TargetID[np.nonzero(y_class_TargetID)]
        print(y_class_TargetID_nonzero)
        actives=np.count_nonzero(y_class_TargetID_nonzero==1)
        ratio=actives/y_class_TargetID_nonzero.shape[1]
        ratios.append(ratio)
print(len(ratios))
ratio_np=np.array(ratios)
ratio01=np.sum(ratio_np<0.1)/len(ratios)
ratio001=np.sum(ratio_np<0.01)/len(ratios)

print('01:', ratio01)
print('001:', ratio001)

bioactivities=[]
'''for j in range(0, y_class_chembl.shape[1]):
    print(j)
    y_class_TargetID=y_class_chembl[:, j]
    if y_class_TargetID.size==0:
        pass
    else:
        y_class_TargetID_nonzero=y_class_TargetID[np.nonzero(y_class_TargetID)]
        bioactivities.append(y_class_TargetID_nonzero.shape[1])


print(len(bioactivities))
bioactivities_np=np.arry(bioactivities)'''
'''bioactivities=np.load('/home/rosa/git/SparseChem/examples/chembl/auxiliary_datafiles/bioactvities_ChemblPubChem.npy')
x_val_small=range(0,10001,500)
plt.hist(bioactivities, x_val_small)
plt.xlabel('Number of bioactivities')
plt.ylabel('Number of targets')
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/auxiliary_datafiles/Chembl_PubChemData_small.png')'''
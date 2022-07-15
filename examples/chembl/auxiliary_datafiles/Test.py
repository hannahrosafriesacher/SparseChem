
import sparsechem as sc
import pandas as pd
import numpy as np
import scipy.sparse as sparse


compound_IDs=pd.read_csv('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Large_chembl_29_Uniprot_compoundIDs.csv', names=['index', 'cids'])
compound_IDs_list=compound_IDs['cids'].to_list()

target_ID=pd.read_csv('/home/rosa/git/SparseChem/PubChem_Actives_Inactives/Targets_Uniprot_ChemblIDs.csv', names=['Chembl', 'uniprots'])
target_ID_list_Chembls=target_ID['Chembl']
target_ID_list_Uniprots=target_ID['uniprots'].to_list()

measurements_large=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Large_chembl_29_Uniprot_inactives_thresh.npy')
measurements_large_txt=pd.read_csv('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Large_sparse.csv', sep=',', names=['cmpd_id', 'target_id', 'activity'])


target_id_list=[]
cmpd_id_list=[]
Uniprot_id_list=[]
variable_list=[]
value_list=[measurements_large_txt['activity'].tolist()]
value_list=value_list[0]
print(type(value_list), len(value_list))

compound_lists_index=measurements_large_txt['cmpd_id'].to_list()
targets_lists_index=measurements_large_txt['target_id'].to_list()


for i in range(len(compound_lists_index)):
    cmpd_id_list.append(compound_IDs_list[compound_lists_index[i]])
print(len(cmpd_id_list))

for k in range(len(targets_lists_index)):
    print('k--------------------------', k)
    print(int(targets_lists_index[k]))
    if int(targets_lists_index[k]) <888:
        Uniprot_id_list.append(target_ID_list_Uniprots[targets_lists_index[k]])
        target_id_list.append(target_ID_list_Chembls[targets_lists_index[k]])  
        variable_list.append(5.5)
    if int(targets_lists_index[k]) >887 and int(targets_lists_index[k]) <1776:
        Uniprot_id_list.append(target_ID_list_Uniprots[int(targets_lists_index[(k)])-888])
        target_id_list.append(target_ID_list_Chembls[int(targets_lists_index[(k)])-888])
        variable_list.append(6.5)
    if int(targets_lists_index[k]) >1775 and int(targets_lists_index[k]) <2664:
        Uniprot_id_list.append(target_ID_list_Uniprots[int(targets_lists_index[(k)])-1776])
        target_id_list.append(target_ID_list_Chembls[int(targets_lists_index[(k)])-1776])
        variable_list.append(7.5)
    if int(targets_lists_index[k]) >2663:
        Uniprot_id_list.append(target_ID_list_Uniprots[int(targets_lists_index[(k)])-2664])
        target_id_list.append(target_ID_list_Chembls[int(targets_lists_index[(k)])-2664])
        variable_list.append(8.5)

print(len(Uniprot_id_list))
print(len(target_id_list))
print(len(variable_list))
print(len(value_list))

df=pd.DataFrame(zip(target_id_list, cmpd_id_list, Uniprot_id_list, variable_list, value_list), columns=['target_id', 'cmpd_id', 'UniprotID', 'variable', 'value'])
print(df)
df.to_csv('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/Large_chembl_29_Uniprot_inactives_thresh_csr.csv')
import sparsechem as sc
import argparse

##This file retrieves maxAUCROC/maxAUCPR/lastAUCROC/lastAUCPR values of the model from the json file
parser = argparse.ArgumentParser(description="Selecting highest und last AUCROC/AUCPR values")
parser.add_argument("--file", help="json file of the model)", type=str, default=None)
parser.add_argument('--targetID', type=int)
args = parser.parse_args()

##load json file of the models
re=sc.load_results(args.file)

##select sub-dictionaries in json file
val= re['validation']
clas=val['classification']
print(val)
print('-----------------------------------------------------')
print(clas)
print('-----------------------------------------------------')

##select ROCAUC/ROCPR rows and select maximum
clas_roc_auc=clas.iloc[:,0]
clas_auc_pr=clas.iloc[:,0]
max_roc_auc=clas_roc_auc.max()
max_auc_pr=clas_auc_pr.max()

print('AUCROC specific target', clas_roc_auc.iloc[args.targetID])
print('-----------------------------------------------------')

##retrieve lastAUCROC/lastAUCPR(=average)
clas_ag=val['classification_agg']
clas_ag_roc_auc=clas_ag.iloc[0]
clas_ag_auc_pr=clas_ag.iloc[1]

##print values
print('max_roc_auc', max_roc_auc,'max_auc_pr', max_auc_pr, 'clas_ag_roc_auc',  clas_ag_roc_auc, 'clas_ag_auc_pr', clas_ag_auc_pr)

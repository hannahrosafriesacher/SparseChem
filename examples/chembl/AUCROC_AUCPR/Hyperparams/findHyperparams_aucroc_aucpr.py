import pandas as pd
import numpy as np

##This file takes results_hyperParams(output of aucroc_aucpr) as input, retrieves the maxima of AUCROCmax/AUCPRmax/AUCROClast/AUCPRlast values and obtains the Hyperparamaters

##initialize Pandas DataFrame
values_df=pd.DataFrame(columns=['hiddenSizes', 'lastDropout', 'weightDecay', 'AUCROC_max', 'AUCPR_max','AUCROC_last', 'AUCPR_last'])

##load data
input=open('./AUCROC_AUCPR/results_extract_aucroc_aucpr', 'r').readlines()
for l in range(0, len(input), 2):
    hyperParams=input[l].split()
    values=input[l+1].split()
    values_df.loc[len(values_df)]=[hyperParams[0], hyperParams[1], hyperParams[2], float(values[0]),float(values[1]),float(values[2]),float(values[3])]

##retrieves maxima for AUCROCmax/AUCPRmax/AUCROClast/AUCPRlast
AUCROC_maxmax=values_df['AUCROC_max'].max()
AUCPR_maxmax=values_df['AUCPR_max'].max()
AUCROC_lastmax=values_df['AUCROC_last'].max()
AUCPR_lastmax=values_df['AUCPR_last'].max()

##retrieves indeces of  maxima for AUCROCmax/AUCPRmax/AUCROClast/AUCPRlast
AUCROC_maxmax_indx=values_df['AUCROC_max'].idxmax()
AUCPR_maxmax_indx=values_df['AUCPR_max'].idxmax()
AUCROC_lastmax_indx=values_df['AUCROC_last'].idxmax()
AUCPR_lastmax_indx=values_df['AUCPR_last'].idxmax()

##retrieves Hyperparameters of maxima for AUCROCmax/AUCPRmax/AUCROClast/AUCPRlast
AUCROC_maxmax_HyperParams=values_df.iloc[AUCROC_maxmax_indx,0:3]
AUCPR_maxmax_HyperParams=values_df.iloc[AUCPR_maxmax_indx,0:3]
AUCROC_lastmax_HyperParams=values_df.iloc[AUCROC_lastmax_indx,0:3]
AUCPR_lastmax_HyperParams=values_df.iloc[AUCPR_lastmax_indx,0:3]

##print pretty
print(values_df)

print('------------------------------------------------------------------------')
print('Maximum Value AUCROCmax: ', AUCROC_maxmax)
print('Index Maximum Value AUCROCmax: ', AUCROC_maxmax_indx)
print('HyperParams- Maximum Value AUCROCmax: ')
print(AUCROC_maxmax_HyperParams)
print('------------------------------------------------------------------------')

print('Maximum Value AUCPRmax: ', AUCPR_maxmax)
print('Index Maximum Value AUCPRmax: ',AUCPR_maxmax_indx)
print('HyperParams- Index Maximum Value AUCPRmax: ')
print(AUCPR_maxmax_HyperParams)
print('------------------------------------------------------------------------')

print('Maximum Value AUCROClast: ', AUCROC_lastmax)
print('Index Maximum Value AUCROClast: ', AUCROC_lastmax_indx)
print('HyperParams- Index Maximum Value AUCROClast: ')
print(AUCROC_lastmax_HyperParams)
print('------------------------------------------------------------------------')

print('Maximum Value AUCPRlast: ', AUCPR_lastmax)
print('Index Maximum Value AUCPRlast: ', AUCPR_lastmax_indx)
print('HyperParams- Index Maximum Value AUCPRlast: ')
print(AUCPR_lastmax_HyperParams)






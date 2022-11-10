import numpy as np
import torch
import scipy.sparse
import sparsechem as sc
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from tabulate import tabulate
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Training a multi-task model.")
parser.add_argument("--TargetID", type=str, default=None)
args = parser.parse_args()
TargetID=args.TargetID
print('Target_ID' +str(TargetID)+'.................................................................................................')

hidden_sizes_list=[7]
lr_list=[2e-4]
dropout_list=[0.7]
weight_decay_list=[ 0.01]
te_fold=0
epoch_number=150
num_output_features=1
batch_size=100

#load Datasets
#X_singleTask=scipy.sparse.csr_matrix(np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/X_'+str(TargetID)+'_reduced_folded4.npy', allow_pickle=True))
X_singleTask=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/X_'+str(TargetID)+'_reduced.npy')
Y_singleTask=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/y_'+str(TargetID)+'_reduced.npy')
folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_singleTask/folding_'+str(TargetID)+'_reduced.npy')

#Training Data
X_singleTask_filtered=X_singleTask[np.logical_and(folding!=1, folding!=0)].todense()
Y_singleTask_filtered=Y_singleTask[np.logical_and(folding!=1, folding!=0)].todense()
Y_singleTask_filtered[Y_singleTask_filtered==-1]=0

#Test Data
X_test_np=X_singleTask[folding==te_fold].todense()
Y_test_np=Y_singleTask[folding==te_fold].todense()
Y_test_np[Y_test_np==-1]=0

#to Torch
X_train=torch.from_numpy(X_singleTask_filtered).float()
Y_train=torch.from_numpy(Y_singleTask_filtered)
X_test=torch.from_numpy(X_test_np).float()
Y_test=torch.from_numpy(Y_test_np)
num_input_features=X_singleTask_filtered.shape[1]
i=0

for i in range(len(hidden_sizes_list)):
    hidden_sizes=hidden_sizes_list[i]
    m=0
    results_test_loss_last_epoch=[]
    for m in range(len(weight_decay_list)):
        weight_decay=weight_decay_list[m]
        j=0
        results_test_loss_last_epoch_single=[]
        for j in range(len(dropout_list)):
            dropout=dropout_list[j]
            results_test_loss_all_epochs=[]
            results_training_loss_all_epochs=[]
            #Model
            class Net(torch.nn.Module):
                def __init__(self, input_features, hidden_sizes, output_features, dropout):
                    super(Net,self).__init__()
                    self.input_features=input_features
                    self.hidden_sizes=hidden_sizes
                    self.output_features=output_features
                    self.dropout=dropout
                    self.net = torch.nn.Sequential(
                        #SparseLinearLayer,
                        torch.nn.Linear(in_features=input_features, out_features=hidden_sizes),
                        #Relu,
                        torch.nn.ReLU(),
                        #Dropout
                        torch.nn.Dropout(p=dropout),
                        #Linear,
                        torch.nn.Linear(hidden_sizes, output_features),
                    )
                    
                def forward(self, x):
                    return self.net(x)
            #training loop
            net=Net(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=num_output_features, dropout=dropout)
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(net.parameters(), lr=2e-4, weight_decay=weight_decay)
            results_loss_test_all_epochs=[]
            results_loss_training_all_epochs_training=[]
            for epoch in range(epoch_number):  # loop over the dataset multiple times 
                permutation = torch.randperm(X_train.size()[0])
                for i in range(0,X_train.size()[0], batch_size):
                    optimizer.zero_grad()
                    # get the inputs; data is a list of [inputs, labels]
                    indices = permutation[i:i+batch_size]
                    inputs, labels = X_train[indices], Y_train[indices]        
                    # forward + backward + optimizer
                    net.eval()
                    outputs = net(inputs)
                    net.train()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                               
                #get loss of each epoch for plotting convergence
                net.eval()
                pred_train=net(X_train)
                loss_train=criterion(pred_train, Y_train).detach().item()
                pred=net(X_test)
                loss_test=criterion(pred, Y_test).detach().item()
                #print('----Test_set----')
                #print('Epoch: ', epoch, ', Loss: ', loss_test)    
                results_loss_test_all_epochs.append(loss_test)
                results_loss_training_all_epochs_training.append(loss_train)
                if len(results_loss_test_all_epochs)>2:
                    if results_loss_test_all_epochs[epoch]>results_loss_test_all_epochs[epoch-1]:
                        break  
            
            results_test_loss_all_epochs.append(results_loss_test_all_epochs)
            results_training_loss_all_epochs.append(results_loss_training_all_epochs_training)

            #getting list with losses of each params combiation (take loss of last epoc)               
            net.eval()
            pred=net(X_test)
            loss_test=criterion(pred, Y_test).detach().item()
            results_test_loss_last_epoch_single.append(loss_test)
                
        results_test_loss_last_epoch.append(results_test_loss_last_epoch_single)
    print('-------------- HIDDENSIZE: '+str(hidden_sizes)+'-----------')
    #print('---------------WEIGHT_DECAY: '+str(weight_decay)+'---------')
    #print(pd.DataFrame(results_test_loss_last_epoch, index=weight_decay_list, columns= dropout_list))
    
    #Plot convergence
    plt.clf()
    for l in range(len(results_test_loss_all_epochs)):
        print(l)
        plt.plot(results_test_loss_all_epochs[l])
        plt.plot(results_training_loss_all_epochs[l], linestyle='dashed')
    #plt.legend(dropout_list)
    plt.savefig('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/'+str(TargetID)+'_Convergence_Plots/opt_Convergence_on_TrainingTestLoss_HS_'+str(hidden_sizes)+'_WD_'+str(weight_decay)+'_LR_'+str(2e-4)+'.png')
    

#Function Accuracy
def check_accuracy(X_test_data, Y_test_data, model):
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for i in range(X_test_data.size()[0]):
            data = X_test_data[i]
            labels = Y_test_data[i]

            net.eval()
            predictions = torch.sigmoid(model(data))
            predictions_round=torch.round(predictions)
            num_correct += (predictions_round == labels).sum()


        return float(num_correct)/float(X_test_data.size()[0])
#check accuracy of test fold
accuracy=check_accuracy(X_test, Y_test, net)
print('Accuracy', accuracy)

#Predict test fold
net.eval()
Y_hat=torch.sigmoid(net(X_test))
#np.save('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SM_SingleTask/SingleTask_'+str(TargetID)+'_hiddenSizes_'+str(hidden_sizes)+'_te_fold_'+str(te_fold)+'_lr_'+str(lr)+'_ep_'+str(epoch_number)+'_batch_size'+str(batch_size)+ '_dropout' + str(dropout)+'_class.npy', Y_hat.detach().numpy())


#-------------------------------------------------------AUC VALUES; CALIBRATION--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
y_class=Y_test_np
y_hat=np.asarray(Y_hat.detach())
#!!!!!!
#Y_hat=torch.sigmoid(net(X_train))
#y_class=Y_singleTask_filtered
#y_hat=np.asarray(Y_hat.detach())
#_________________________________________ROC_AUC, PR_AUC values_________________________________________________________
AUC=roc_auc_score(y_class, y_hat)
precision, recall, thresholds = precision_recall_curve(y_class, y_hat)
PR=auc(recall, precision)
print('AUC:', AUC)
print('PR:', PR)

#_________________________________________ACE, ECE values_________________________________________________________________
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
    clas.extend(split(y_class,np.logical_and(y_hat>=values[j], y_hat<values[j+1])).flatten())
    prob.append(split(y_hat,np.logical_and(y_hat>=values[j], y_hat<values[j+1])).flatten())

    j+=1
#Obtain positive ratio (=acc calculated from true values) and 
# probablity mean (=conf calculated from predictions) for each split
for k in range(10):
    acc.append(posRatio(clas[k], 1))
    conf.append(ProbMean(prob[k], 0, k))
    k+=1

#obtain ACE for this specific target:
ece=0
for i in range(10):
    ece+=(np.abs(np.array(acc[i])-np.array(conf[i]))*clas[i].shape[1])
    #      |               acc(b)-         conf(n)| * nb
ece=ece/y_class.shape[0]
#   sumofECE/N
print('ECE of Target', ece)


##-----------------------ACE-----------------------

#sort class and hat file by ascending probablity values in hat file
index_sort_y_hat=np.argsort(y_hat.flatten())
y_hat_sorted=y_hat[index_sort_y_hat]
y_class_sorted=y_class[index_sort_y_hat].flatten().T

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

#obtain ACE for this specific target:
ace=np.sum(np.abs(acc_ace-conf_acc))/10
#     SumOverAllR(|acc(b)-conf(b)|)/R
print('ACE of Target', ace)

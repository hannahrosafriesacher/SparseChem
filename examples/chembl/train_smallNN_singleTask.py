import numpy as np
import torch
import sparsechem as sc
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

hidden_sizes=40
epoch_number=50
lr=1e-6
te_fold=0
num_output_features=1
batch_size=200

#load Datasets
X_singleTask=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/X_1482_reduced.npy')
Y_singleTask=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/y_1482_reduced.npy')
folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/folding_1482_reduced.npy')

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
#Model
class Net(torch.nn.Module):
    def __init__(self, input_features, hidden_sizes, output_features):
        super(Net,self).__init__()
        self.input_features=input_features
        self.hidden_sizes=hidden_sizes
        self.output_features=output_features
        self.net = torch.nn.Sequential(
            #SparseLinearLayer,
            torch.nn.Linear(in_features=input_features, out_features=hidden_sizes),
            #Relu,
            torch.nn.ReLU(),
            #Linear,
            torch.nn.Linear(hidden_sizes, output_features)
            
        )
    
    def forward(self, x):
        return self.net(x)


#Function Accuracy
def check_accuracy(X_test_data, Y_test_data, model):
    num_correct = 0
    model.eval()
    with torch.no_grad():
        for i in range(X_test_data.size()[0]):
            data = X_test_data[i]
            labels = Y_test_data[i]

            predictions = torch.sigmoid(model(data))
            predictions_round=torch.round(predictions)
            num_correct += (predictions_round == labels).sum()


        return float(num_correct)/float(X_test_data.size()[0])

#training loop for different hidden sizes
net=Net(hidden_sizes=hidden_sizes, input_features=num_input_features, output_features=num_output_features)
criterion = torch.nn.BCEWithLogitsLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=lr)
for epoch in range(epoch_number):  # loop over the dataset multiple times
    running_loss = 0.0
        
    permutation = torch.randperm(X_train.size()[0])

    for i in range(0,X_train.size()[0], batch_size):
        optimizer.zero_grad()
        # get the inputs; data is a list of [inputs, labels]
        indices = permutation[i:i+batch_size]
        inputs, labels = X_train[indices], Y_train[indices]        
        # forward + backward + optimizer
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('running loss epoch '+str(epoch)+':', running_loss/(X_train.size()[0]/batch_size))

accuracy=check_accuracy(X_test, Y_test, net)
print('Accuracy', accuracy)
Y_hat=torch.sigmoid(net(X_test))
np.save('/home/rosa/git/SparseChem/examples/chembl/predictions/model_SingleTask/SingleTask_1482_hiddenSizes_'+str(hidden_sizes)+'_te_fold_'+str(te_fold)+'_lr_'+str(lr)+'_ep_'+str(epoch_number)+'_batch_size'+str(batch_size)+'_class.npy', Y_hat.detach().numpy())

y_class=Y_test_np
y_hat=np.asarray(Y_hat.detach())

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

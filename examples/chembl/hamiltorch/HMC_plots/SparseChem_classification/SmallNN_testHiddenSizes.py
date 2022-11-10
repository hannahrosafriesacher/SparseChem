import numpy as np
import torch
import sparsechem as sc
import torch.optim as optim

hidden_sizes=[2,3,4,5,6,7,8,9,10]
epoch_number=50
#load Datasets
X_singleTask=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/X_1482_reduced.npy')
Y_singleTask=sc.load_sparse('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/y_1482_reduced.npy')
folding=np.load('/home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/folding_1482_reduced.npy')

#Training Data
X_singleTask_filtered=X_singleTask[np.logical_and(folding!=1, folding!=0)].todense()
Y_singleTask_filtered=Y_singleTask[np.logical_and(folding!=1, folding!=0)].todense()
Y_singleTask_filtered[Y_singleTask_filtered==-1]=0

#Test Data
X_test_np=X_singleTask[folding==1].todense()
Y_test_np=Y_singleTask[folding==1].todense()
Y_test_np[Y_test_np==-1]=0


#to Torch
X_train=torch.from_numpy(X_singleTask_filtered).float()
Y_train=torch.from_numpy(Y_singleTask_filtered)
X_test=torch.from_numpy(X_test_np).float()
Y_test=torch.from_numpy(Y_test_np)
print(X_train.size(), Y_train.size())

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

num_input_features=X_singleTask_filtered.shape[1]
num_output_features=1
totalLossPerHiddenSize=[]
EpochLossPerHiddenSize=[]
batch_size=200

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
for hidden_size in range(len(hidden_sizes)):
    net=Net(hidden_sizes=hidden_sizes[hidden_size], input_features=num_input_features, output_features=num_output_features)
    criterion = torch.nn.BCEWithLogitsLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-1)
    hidden_size_loss=[]
    for epoch in range(epoch_number):  # loop over the dataset multiple times
        running_loss = 0.0
        
        permutation = torch.randperm(X_train.size()[0])

        for i in range(0,X_train.size()[0], batch_size):
            optimizer.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            indices = permutation[i:i+batch_size]
            inputs, labels = X_train[indices], Y_train[indices]        

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('running loss: ', running_loss/(X_train.size()[0]/batch_size))
        hidden_size_loss.append(running_loss)

    print('------------hiddenSize-----------:', hidden_sizes[hidden_size])
    print('Loss:', hidden_size_loss[-1]/(X_train.size()[0]/batch_size))

    accuracy=check_accuracy(X_test, Y_test, net)
    print('Accuracy', accuracy)

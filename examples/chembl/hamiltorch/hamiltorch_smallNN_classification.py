#HMC on small NN
import torch.nn as nn
import torch
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as TensorDataset
import torch.nn.functional as F
import hamiltorch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

print('torch:', torch.__version__)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--tau_list', type=float, required=True)
parser.add_argument('--stepsize', type=float, required=True)
parser.add_argument('--StepsPerSample', type=int, required=True)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']='2'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#specify tau_out
tau_out=1

#Test set
#x
torch.manual_seed(100)
x_train=torch.rand(20,1)*10-5
torch.manual_seed(120)
sd=torch.randn(20,1)
#y
y_train=0.5*x_train+2+sd
#y_train to classification
y_train_bool=torch.where(y_train<1.5, 0, 1).float().to(device)


#Training set
#x
x_test=torch.linspace(-5, 5, 100).unsqueeze(dim=1).to(device)
#y_test=torch.sin(x_test*4)
#y
y_test=0.5*x_test+2
y_test_bool=torch.where(y_test<1.5, 0, 1).float().to(device)

#plt.scatter(y_train, y_train, c=y_train_bool)
#plt.savefig('Test.png')

#train smallNN
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net,self).__init__()

        self.hidden1=nn.Linear(1,hidden_size)
        #self.hidden2=nn.Linear(hidden_size,hidden_size)
        #self.hidden3=nn.Linear(hidden_size,hidden_size)
        self.hidden4=nn.Linear(hidden_size,1)

    def forward(self,x):
        x=F.relu(self.hidden1(x))
        #x=F.relu(self.hidden2(x))
        #x=F.relu(self.hidden3(x))
        x=self.hidden4(x)
        return x
hidden_size=3
net=Net(hidden_size=hidden_size)


#specify tau_list
tau_list = []
tau = args.tau_list
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

#train HMC
hamiltorch.set_random_seed(123)
params_init=hamiltorch.util.flatten(net).to(device).clone() 
step_size=args.stepsize
num_samples=120
num_steps_per_sample=args.StepsPerSample
burn_in=10
model_loss='binary_class_linear_output'
params_hmc=hamiltorch.sample_model(net, x_train, y_train_bool, params_init=params_init,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, tau_out=tau_out, model_loss=model_loss, burn=burn_in, tau_list=tau_list)
params_hmc_gpu=[ll.to(device) for ll in params_hmc[1:]]


#predict test set
pred_list, log_prob_list=hamiltorch.predict_model(net, x=x_test, y=y_test_bool, samples=params_hmc_gpu, model_loss=model_loss, tau_out=tau_out)
pred_list=pred_list.detach().cpu()
print(pred_list)
print(type(pred_list))
#pred_list=torch.stack(pred_list)
print(pred_list.size)
y_test_bool=y_test_bool.cpu()

#Accuracy and Negative Loss for each sample:
acc_mean = torch.zeros( len(pred_list)-1)
acc_sample= torch.zeros( len(pred_list)-1)
loss = torch.zeros( len(pred_list)-1)
criterion=nn.BCEWithLogitsLoss(reduction='sum')

for s in range(1, len(pred_list)):
    #ACCURACY
    # take mean of predictions #1-s, round predictions to 0/1 (=label)
    pred_mean=torch.where(torch.sigmoid(pred_list[:s]).mean(0)<0.5, 0, 1).float()
    pred_now_sig=torch.round(torch.sigmoid(pred_list[s]))
    #boolean array: does label correspond to true value in y_test?
    bo_mean=(pred_mean==y_test_bool).flatten()
    bo_sample=(pred_now_sig==y_test_bool).flatten()
    #acc  = TP+TN (#correctPred)/ TP+TN+FP+FN (all datapoints in )         
    acc_mean[s-1]=bo_mean.sum().float()/y_test_bool.shape[0]
    acc_sample[s-1]=bo_sample.sum().float()/y_test_bool.shape[0]

    #LOSS
    pred_now=pred_list[s]
    #calculate loss for current s
    loss[s-1]=criterion(pred_now, y_test_bool)

#AUC AND LOSS PLOT
fig, axs=plt.subplots(2, 1, figsize=(9,9))
fig.suptitle('NumOfSamples: ' +str(num_samples) + '/StepsPerSample: ' + str(num_steps_per_sample) + '/Stepsize: ' + str(step_size) + '\n BurnIn: ' + str(burn_in) + '/tau_I: ' +str(tau))
lim=num_samples-burn_in
axs[0].plot(acc_mean)
axs[0].plot(acc_sample)
axs[0].grid()
axs[0].set_xlabel('Iteration number')
axs[0].set_ylabel('Sample accuracy')
axs[0].tick_params(labelsize=15)
axs[0].set_xlim(0,lim)

axs[1].plot(loss)
axs[1].grid()
axs[1].set_xlabel('Iteration number')
axs[1].set_ylabel('Negative Log-Likelihood')
axs[1].set_xlim(0,lim)
axs[1].tick_params(labelsize=15)
plt.savefig('./HMC_plots/Classification_SmallNN/smallNNClassification__stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(hidden_size)+'_numTraining'+str(x_train.shape[0])+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'.png')
'''
#AUTOCORRELATION PLOZ
autocorr_list=[]
for i in range(len(params_hmc_gpu)):
    x=params_hmc_gpu[i].cpu().numpy()
    autocorr_list.append(x)

autocorr=np.array(autocorr_list)
print(autocorr.shape)

#autocorrrelation plot
for i in range(autocorr.shape[1]):
    index=np.random.choice(autocorr.shape[1], 10, replace=False)
print(autocorr.shape)
print(index)
for i in range(index.shape[0]):
    plt.acorr(autocorr[:,index[i]], usevlines=False, linestyle='solid', marker='.', maxlags=num_samples-burn_in-2)
plt.xlim([0,num_samples-burn_in-2])

plt.savefig('./HMC_plots/Classification_SmallNN/(Autocorr_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(hidden_size)+'_numTraining'+str(x_train.shape[0])+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'.png')
'''
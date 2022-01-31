import torch.nn as nn
import torch
import torch.nn.functional as F
import hamiltorch
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='2'


x_train=torch.rand(100,1)*5-2
y_train=torch.sin(x_train*3)/x_train
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_test=torch.linspace(-2, 3, 100).unsqueeze(dim=1)
y_test=torch.sin(x_test*3)/x_test


class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net,self).__init__()

        self.hidden1=nn.Linear(1,hidden_size)
        self.hidden2=nn.Linear(hidden_size,hidden_size)
        #self.hidden3=nn.Linear(5,5)
        self.predict=nn.Linear(hidden_size,1)

    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        #x=F.relu(self.hidden3(x))
        x=self.predict(x)
        return x
hidden_size=10
net=Net(hidden_size=hidden_size)

hamiltorch.set_random_seed(123)
params_init=hamiltorch.util.flatten(net).to(device).clone()
step_size=0.001
num_samples=120
num_steps_per_sample=100
burn_in=10
model_loss='regression'
tau_out=1
params_hmc=hamiltorch.sample_model(net, x_train, y_train, params_init=params_init,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, tau_out=tau_out, model_loss=model_loss, burn=burn_in)
print(params_hmc)

params_hmc_gpu=[ll.to(device) for ll in params_hmc[1:]]

predictions, log_probs=hamiltorch.predict_model(net, x=x_test.to(device), y=y_test.to(device), samples=params_hmc_gpu, model_loss=model_loss, tau_out=tau_out)
predictions=predictions.detach().cpu()
mean=predictions.mean(0).squeeze()
std=predictions.std(0).squeeze()

plt.plot(x_train, y_train, 'or')
plt.plot(x_test, mean, '-', color='grey')

for i in range(num_samples-burn_in-1):
    plt.plot(x_test, predictions[i].squeeze(), alpha=.2)
plt.fill_between(x_test.squeeze(),(mean-std),(mean+std), color='grey', alpha=0.2)
plt.savefig('./HMC_plots/HMC_smallNN_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(hidden_size)+'numLayers'+str(2)+'_numTraining'+str(x_train.shape[0])+'.pdf')

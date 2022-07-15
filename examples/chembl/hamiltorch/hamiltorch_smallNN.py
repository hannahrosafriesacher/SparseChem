#HMC on small NN
import torch.nn as nn
import torch
import torch.nn.functional as F
import hamiltorch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="")
#parser.add_argument("--variance", type=float, required=True)
#parser.add_argument('--tau_list', type=float, required=True)
parser.add_argument('--stepsize', type=float, required=True)
parser.add_argument('--StepsPerSample', type=int, required=True)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']='2'

#specify noise and tau_out
#var=args.variance
tau_out=1

#Test set
torch.manual_seed(101)
x_train=torch.rand(4,1)*5-2
#torch.manual_seed(120)
#sd=(var**0.5)*torch.randn(4,1)
y_train=torch.sin(x_train*3)/x_train
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Training set
x_test=torch.linspace(-2, 3, 2).unsqueeze(dim=1)
y_test=torch.sin(x_test*3)/x_test

print('...............', args.stepsize, args.StepsPerSample)

#train smallNN
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

#specify tau_list
#tau_list = []
#tau = args.tau_list
#for w in net.parameters():
#    tau_list.append(tau)
#tau_list = torch.tensor(tau_list).to(device)

#train HMC
hamiltorch.set_random_seed(123)
params_init=hamiltorch.util.flatten(net).to(device).clone()
step_size=args.stepsize
num_samples=120
num_steps_per_sample=args.StepsPerSample
burn_in=10
model_loss='regression'
params_hmc=hamiltorch.sample_model(net, x_train, y_train, params_init=params_init,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, tau_out=tau_out, model_loss=model_loss, burn=burn_in)
params_hmc_gpu=[ll.to(device) for ll in params_hmc[1:]]

autocorr_list=[]
for i in range(len(params_hmc_gpu)):
    x=params_hmc_gpu[i].cpu().numpy()
    autocorr_list.append(x)

autocorr=np.array(autocorr_list)
print(autocorr.shape)

#autocorrrelation plot
#for i in range(autocorr.shape[1]):
#index=np.random.choice(autocorr.shape[1], 10, replace=False)
#print(autocorr.shape)
#print(index)
#for i in range(index.shape[0]):
#    plt.acorr(autocorr[:,index[i]], usevlines=False, linestyle='solid', marker='.', maxlags=num_samples-burn_in-2)
#plt.xlim([0,num_samples-burn_in-2])

#plt.savefig('./HMC_plots/Autocorr_smallNN_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(hidden_size)+'numLayers'+str(2)+'_numTraining'+str(x_train.shape[0])+'.png')
    
#predict test set
predictions, log_probs=hamiltorch.predict_model(net, x=x_test.to(device), y=y_test.to(device), samples=params_hmc_gpu, model_loss=model_loss, tau_out=tau_out)
predictions=predictions.detach().cpu()
mean=predictions.mean(0).squeeze()
std=predictions.std(0).squeeze()

#tau=1
#var=1/tau_out
#plot models
plt.plot(x_train, y_train, 'or')
plt.plot(x_test, mean, '-', color='grey')


#for i in range(num_samples-burn_in-1):
#    plt.plot(x_test, predictions[i].squeeze(), alpha=.2)
plt.fill_between(x_test.squeeze(),(mean-std),(mean+std), color='grey', alpha=0.2)
#plt.title('tau_out: '+str(round(tau_out, 2))+'\n tau_I: '+str(tau)+ '\n lambda: '+ str( round((tau/tau_out), 3)))
#plt.savefig('./HMC_plots/Regression_SmallNN/TEST_HMC_smallNN_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(hidden_size)+'numLayers'+str(2)+'_numTraining'+str(x_train.shape[0])+'_var'+ str(round(var, 2))+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'.png')
plt.savefig('./HMC_plots/Regression_SmallNN/TEST.png')

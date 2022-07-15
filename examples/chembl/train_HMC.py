# Copyright (c) 2020 KU Leuven
import sparsechem as sc
import hamiltorch
import scipy.sparse
import numpy as np
import pandas as pd
import torch
import argparse
import os
import os.path
from sparsechem import Nothing
from torch.utils.tensorboard import SummaryWriter
import math
import torch.nn as nn
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Training a multi-task model.")
#tau_list, stepsize, burn_in, steps_per_sample, num of Samples, output, input features
parser.add_argument("--x", help="Descriptor file (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_class", "--y", "--y_classification", help="Activity file (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_regr", "--y_regression", help="Activity file (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--weights_class", "--task_weights", "--weights_classification", help="CSV file with columns task_id, training_weight, aggregation_weight, task_type (for classification tasks)", type=str, default=None)
parser.add_argument("--folding", help="Folding file (npy)", type=str, required=True)
parser.add_argument("--fold_va", help="Validation fold number", type=int, default=0)
parser.add_argument("--fold_te", help="Test fold number (removed from dataset)", type=int, default=None)
parser.add_argument("--y_censor", help="Censor mask for regression (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--batch_ratio", help="Batch ratio", type=float, default=0.2)
parser.add_argument("--internal_batch_max", help="Maximum size of the internal batch", type=int, default=None)
parser.add_argument("--tau_list", help="taulist", type=float, required=True)
parser.add_argument("--tau_out", help="tauout", type=float, required=True)
parser.add_argument("--stepsize", help="StepSize", type=float, required=True)
parser.add_argument("--num_samples", help="Number of Samples", type=int, required=True)
parser.add_argument("--StepsPerSample", help="StepsPerSample", type=int, required=True)
parser.add_argument("--hidden_sizes", help="HiddenSize", type=int, required=True)
parser.add_argument("--burn_in", help="brun_in", type=int, required=True)

parser.add_argument("--input_transform", help="Transformation to apply to inputs", type=str, default="none", choices=["binarize", "none", "tanh", "log1p"])
parser.add_argument("--min_samples_class", help="Minimum number samples in each class and in each fold for AUC calculation (only used if aggregation_weight is not provided in --weights_class)", type=int, default=5)
parser.add_argument("--run_name", help="Run name for results", type=str, default=None)

args = parser.parse_args()

ecfp     = sc.load_sparse(args.x)
y_class  = sc.load_sparse(args.y_class)
y_regr   = sc.load_sparse(args.y_regr)
y_censor = sc.load_sparse(args.y_censor)
print('ECFP:', ecfp.shape)
print('Y_class:', y_class.shape)

folding = np.load(args.folding)
assert ecfp.shape[0] == folding.shape[0], "x and folding must have same number of rows"
print('folding:', folding.shape)

if y_regr is None:
    y_regr  = scipy.sparse.csr_matrix((ecfp.shape[0], 0))
if y_censor is None:
    y_censor = scipy.sparse.csr_matrix(y_regr.shape)

## Loading task weights
tasks_class = sc.load_task_weights(args.weights_class, y=y_class, label="y_class")

## Input transformation
ecfp = sc.fold_transform_inputs(ecfp,transform=args.input_transform)

num_pos    = np.array((y_class == +1).sum(0)).flatten()
num_neg    = np.array((y_class == -1).sum(0)).flatten()
num_class  = np.array((y_class != 0).sum(0)).flatten()
if (num_class != num_pos + num_neg).any():
    raise ValueError("For classification all y values (--y_class/--y) must be 1 or -1.")

## using min_samples rule
fold_pos, fold_neg = sc.class_fold_counts(y_class, folding)
n = args.min_samples_class
tasks_class.aggregation_weight = ((fold_pos >= n).all(0) & (fold_neg >= n)).all(0).astype(np.float64)

print(f"Input dimension: {ecfp.shape[1]}")
print(f"#samples:        {ecfp.shape[0]}")
print(f"#classification tasks:  {y_class.shape[1]}")


if args.fold_te is not None and args.fold_te >= 0:
    ## removing test data
    assert args.fold_te != args.fold_va, "fold_va and fold_te must not be equal."
    keep    = folding != args.fold_te
    ecfp    = ecfp[keep]
    y_class = y_class[keep]
    folding = folding[keep]

fold_va = args.fold_va
idx_te=np.nonzero(np.load(args.folding)==args.fold_te)[0]
print(idx_te.shape)
idx_tr  = np.where(folding != fold_va)[0]
print(idx_tr.shape)
idx_va  = np.where(folding == fold_va)[0]
print(idx_va.shape)
y_regr  = y_regr[keep]

y_class_te=sc.load_sparse(args.y_class)[idx_te]
y_class_tr = y_class[idx_tr]
y_class_va = y_class[idx_va]
y_regr_tr  = y_regr[idx_tr]
y_regr_va  = y_regr[idx_va]
y_censor_tr = y_censor[idx_tr]
y_censor_va = y_censor[idx_va]


num_pos_va  = np.array((y_class_va == +1).sum(0)).flatten()
num_neg_va  = np.array((y_class_va == -1).sum(0)).flatten()

batch_size  = int(np.ceil(args.batch_ratio * idx_tr.shape[0]))
num_int_batches = 1

if args.internal_batch_max is not None:
    if args.internal_batch_max < batch_size:
        num_int_batches = int(np.ceil(batch_size / args.internal_batch_max))
        batch_size      = int(np.ceil(batch_size / num_int_batches))
print(f"#internal batch size:   {batch_size}")
x_train=ecfp[idx_tr].todense()
y_train=y_class_tr.todense()
x_test=sc.load_sparse(args.x)[idx_te].todense()
y_test=y_class_te.todense()

('X_train:', x_train.shape)
('y_train:', y_train.shape)
('x_test:', x_test.shape)
('y_test:', y_test.shape)

dataset_tr = sc.ClassRegrSparseDataset(x=ecfp[idx_tr], y_class=y_class_tr, y_regr=y_regr_tr, y_censor=y_censor_tr)
dataset_va = sc.ClassRegrSparseDataset(x=ecfp[idx_va], y_class=y_class_va, y_regr=y_regr_va, y_censor=y_censor_va)
os.environ['CUDA_VISIBLE_DEVICES']='2'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#MODEL
x_trainTorch= torch.tensor(x_train).to(device)
y_trainTorch=torch.tensor(y_train).to(device)
print(x_trainTorch.shape, y_trainTorch.shape)
num_input_features=x_trainTorch.shape[1]
print('input', type(num_input_features))
num_output_features=1
print(num_output_features)

class Net(torch.nn.Module):
    def __init__(self, input_features, hidden_sizes, output_features):
        super(Net,self).__init__()
        self.input_features=input_features
        self.hidden_sizes=hidden_sizes
        self.output_features=output_features
        self.net = torch.nn.Sequential(
            #SparseLinearLayer,
            sc.SparseLinear(input_features, hidden_sizes),
            #Relu,
            torch.nn.ReLU(),
            #Linear,
            torch.nn.Linear(hidden_sizes, output_features)
        )
    
    def forward(self, x):
        return self.net(x)
net=Net(hidden_sizes=args.hidden_sizes, input_features=num_input_features, output_features=num_output_features)
print(type(net))
#specify tau_list
tau_list = []
tau = args.tau_list
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

#train HMC
hamiltorch.set_random_seed(123)
params_init=hamiltorch.util.flatten(net).to(device).clone() 
#print('params_init', params_init.size())
step_size=args.stepsize
num_samples=args.num_samples
num_steps_per_sample=args.StepsPerSample
burn_in=args.burn_in
tau_out=args.tau_out
model_loss='binary_class_linear_output'
params_hmc=hamiltorch.sample_model(net, x=x_trainTorch, y=y_trainTorch, params_init=params_init,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, tau_out=tau_out, model_loss=model_loss, burn=burn_in, tau_list=tau_list)
params_hmc_gpu=[ll.to(device) for ll in params_hmc[1:]]
print(params_hmc_gpu)
#print('params_hmc', len(params_hmc), params_hmc[0].size())
#print(len(params_hmc_gpu), params_hmc_gpu[0].size())

'''# PREDICT

x_testTorch=torch.tensor(x_test).to(device)
y_testTorch=torch.tensor(y_test).to(device)
print('X_test', x_test.shape, 'X_testTorch', x_testTorch.size())
print('Y_test', y_test.shape, 'Y_testTorch', y_testTorch.size())
pred_list, log_prob_list=hamiltorch.predict_model(net, x=x_trainTorch, y=y_trainTorch, samples=params_hmc_gpu, model_loss=model_loss, tau_out=tau_out)
pred_list=pred_list.detach().cpu()
print(pred_list.size())
print(type(pred_list))

y_testTorch=y_testTorch.cpu()
y_trainTorch=y_trainTorch.cpu()

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
    bo_mean=(pred_mean==y_trainTorch).flatten()
    bo_sample=(pred_now_sig==y_trainTorch).flatten()
    #acc  = TP+TN (#correctPred)/ TP+TN+FP+FN (all datapoints in )         
    acc_mean[s-1]=bo_mean.sum().float()/y_trainTorch.shape[0]
    acc_sample[s-1]=bo_sample.sum().float()/y_trainTorch.shape[0]

    #LOSS
    pred_now=pred_list[s]
    #calculate loss for current s
    loss[s-1]=criterion(pred_now, y_trainTorch)


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
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/SparseChem_classification/Target1482/AUC_LOSS_plot_OnTrainingSet_SparseChem_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(args.hidden_sizes)+'_numTraining'+str(x_train.shape[0])+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'.png')'''



#AUTOCORRELATION PLOT

def autocorr(y, lag):
    
    #Calculates autocorrelation coefficient for single lag value
        
    #y: array
    #   Input time series array
    #lag: int, default: 2 
    #     'kth' lag value
        
    #Returns
    #int: autocorrelation coefficient


    y_bar =np.sum(y)/y.shape[0] #y_bar = mean of the time series y
    denominator = sum((y - y_bar) ** 2) #sum of squared differences between y(t) and y_bar
    numerator_p1 = y[lag:] - y_bar #y(t+k)-y_bar: difference between time series (from 'lag' till the end) and y_bar
    numerator_p2 = y[:len(y)-lag] - y_bar #y(t)-y_bar: difference between time series (from the start till lag) and y_bar
    numerator = sum(numerator_p1 * numerator_p2) #sum of y(t)-y_bar and y(t-k)-y_bar
    return numerator/denominator


params_list=[]
for i in range(len(params_hmc_gpu)):
    x=params_hmc_gpu[i].cpu().numpy()
    params_list.append(x)
params=np.array(params_list)
print(type(params), params.shape)
np.save('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/SparseChem_classification/Target1482/Params_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(args.hidden_sizes)+'_numTraining'+str(x_train.shape[0])+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'.npy', params)
index=np.random.choice(params.shape[1], 10, replace=False)

autocorr_whole=[]
for l in range(index.shape[0]):
    autocorr_list_now=[]
    params_now=params[:, index[l]]
    #print('params_now', params_now.shape)
    for i in range(0, params_now.shape[0]-1):
        autocorr_list_now.append(autocorr(params_now, i))
    autocorr_whole.append(autocorr_list_now)

#print(len(autocorr_whole), autocorr_whole)
for j in range(len(autocorr_whole)):
    plt.plot(range(0, len(autocorr_whole[j])),autocorr_whole[j], marker='o')
plt.title('StepSize: ' + str(step_size) + '/ StepsPerSample: ' + str(num_steps_per_sample) + '/ HiddenSizes: ' + str(args.hidden_sizes))
plt.xlabel('lag')
plt.ylabel('autocorrelation')    
plt.xlim([0,num_samples-burn_in])
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/SparseChem_classification/Target1482/Autocorr_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(args.hidden_sizes)+'_numTraining'+str(x_train.shape[0])+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'.png')

#EXAMPLE:
#python train_HMC.py --x /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/X_1482_reduced.npy --y_class /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/y_1482_reduced.npy --folding /home/rosa/git/SparseChem/examples/chembl/files_data_folding_current/datafiles_hmc/folding_1482_reduced.npy --fold_va 1 --fold_te 0 --tau_list 5 --tau_out 0.9 --num_samples 200 --hidden_sizes 5  --burn_in 80 --StepsPerSample 800 --stepsize 1e-7


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
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Training a multi-task model.")
#tau_list, stepsize, burn_in, steps_per_sample, num of Samples, output, input features
parser.add_argument("--x", help="Descriptor file (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--y_class", "--y", "--y_classification", help="Activity file (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--folding", help="Folding file (npy)", type=str, required=True)
parser.add_argument("--fold_va", help="Validation fold number", type=int, default=0)
parser.add_argument("--fold_te", help="Test fold number (removed from dataset)", type=int, default=None)
parser.add_argument("--batch_ratio", help="Batch ratio", type=float, default=0.2)
parser.add_argument("--internal_batch_max", help="Maximum size of the internal batch", type=int, default=None)
parser.add_argument("--tau_list", help="taulist", type=float, required=True)
parser.add_argument("--tau_out", help="tauout", type=float, required=True)
parser.add_argument("--stepsize", help="StepSize", type=float, required=True)
parser.add_argument("--num_samples", help="Number of Samples", type=int, required=True)
parser.add_argument("--StepsPerSample", help="StepsPerSample", type=int, required=True)
parser.add_argument("--hidden_sizes", help="HiddenSize", type=int, required=True)
parser.add_argument("--burn_in", help="brun_in", type=int, required=True)
parser.add_argument("--run_name", help="Run name for results", type=str, default=None)
parser.add_argument("--input_transform", help="Transformation to apply to inputs", type=str, default="none", choices=["binarize", "none", "tanh", "log1p"])
parser.add_argument("--min_samples_class", help="Minimum number samples in each class and in each fold for AUC calculation (only used if aggregation_weight is not provided in --weights_class)", type=int, default=5)
parser.add_argument("--y_regr", "--y_regression", help="Activity file (matrix market, .npy or .npz)", type=str, default=None)
parser.add_argument("--weights_class", "--task_weights", "--weights_classification", help="CSV file with columns task_id, training_weight, aggregation_weight, task_type (for classification tasks)", type=str, default=None)
parser.add_argument("--y_censor", help="Censor mask for regression (matrix market, .npy or .npz)", type=str, default=None)
args = parser.parse_args()

#Variables
os.environ['CUDA_VISIBLE_DEVICES']='0'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tau = args.tau_list
step_size=args.stepsize
num_samples=args.num_samples
num_steps_per_sample=args.StepsPerSample
burn_in=args.burn_in
tau_out=args.tau_out
model_loss='binary_class_linear_output'

#afterBurnin
burn_in_after=100
#load model parameters
filename='Params_stepSize1e-07_numSteps20000_burnIn0_numSamples200_hiddenSize5_numTraining5595_tauOut1.0_tauList0.1_reducedX_10000.npy'
file='/home/rosa/git/SparseChem/examples/chembl/hamiltorch/parameters_HMC/Target1482/'+filename
#load Parameters
params=np.load(file)
print(params.shape)
params_after_burn_in=params[burn_in_after:]
print(params_after_burn_in.shape)
params_hmc_gpu=torch.tensor(params_after_burn_in).to(device)



#load data and files
#ecfp     = sc.load_sparse(args.x)
ecfp=scipy.sparse.csr_matrix(np.load(args.x, allow_pickle=True))
y_class  = sc.load_sparse(args.y_class)
y_regr   = sc.load_sparse(args.y_regr)
y_censor = sc.load_sparse(args.y_censor)

folding = np.load(args.folding)
assert ecfp.shape[0] == folding.shape[0], "x and folding must have same number of rows"

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
    y_regr  = y_regr[keep]

fold_va = args.fold_va
idx_te=np.nonzero(np.load(args.folding)==args.fold_te)[0]
idx_tr  = np.where(folding != fold_va)[0]
idx_va  = np.where(folding == fold_va)[0]

#split y into training, test and validation
y_class_te=sc.load_sparse(args.y_class)[idx_te]
y_class_tr = y_class[idx_tr]
y_class_va = y_class[idx_va]
y_regr_tr  = y_regr[idx_tr]
y_regr_va  = y_regr[idx_va]
y_train=y_class_tr.todense()
y_test=y_class_te.todense()

#split x into Train and Test dataset
x_train=ecfp[idx_tr].todense()
x_test=scipy.sparse.csr_matrix(np.load(args.x, allow_pickle=True))[idx_te].todense()


#To Torch tensor
x_trainTorch_unclipped= torch.tensor(x_train).to(device)
y_trainTorch=torch.tensor(y_train).to(device)
x_trainTorch=torch.clip(x_trainTorch_unclipped, min=0, max=1)

#MODEL
num_input_features=x_trainTorch.shape[1]
num_output_features=1
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
#-------------------------------------------------------------------------------------------
# predict with HMC model


x_testTorch=torch.tensor(x_test).to(device)
y_testTorch=torch.tensor(y_test).to(device)
#pred_list, log_prob_list=hamiltorch.predict_model(net, x=x_testTorch, y=y_testTorch, samples=params_hmc_gpu, model_loss=model_loss, tau_out=tau_out)
pred_list, log_prob_list=hamiltorch.predict_model(net, x=x_trainTorch, y=y_trainTorch, samples=params_hmc_gpu, model_loss=model_loss, tau_out=tau_out)
pred_list=pred_list.detach().cpu()
print(y_test.shape, x_test.shape)
print(pred_list.shape)

y_testTorch=y_testTorch.cpu()
y_trainTorch=y_trainTorch.cpu()

print(pred_list.shape)
pred_list_ensemble=torch.mean(torch.sigmoid(pred_list), 0)
pred_list_ensemble_np=pred_list_ensemble.numpy()
print(pred_list_ensemble_np.shape)

#np.save('/home/rosa/git/SparseChem/examples/chembl/predictions/HMC_singleTask/Target1482/hmc_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in_after)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(args.hidden_sizes)+'_numTraining'+str(x_train.shape[0])+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'_fold_va'+str(args.fold_va)+'_fold_te'+str(args.fold_te)+'_predictionOnTrainingFold.npy', pred_list_ensemble_np)
np.save('/home/rosa/git/SparseChem/examples/chembl/predictions/HMC_singleTask/Target1482/hmc_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in_after)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(args.hidden_sizes)+'_numTraining'+str(x_train.shape[0])+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'_fold_va'+str(args.fold_va)+'_fold_te'+str(args.fold_te)+'.npy', pred_list_ensemble_np)
'''#-----------------------------------------------------------------------------------------------
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
    print(pred_mean.shape, y_testTorch.shape)
    bo_mean=(pred_mean==y_testTorch).flatten()
    bo_sample=(pred_now_sig==y_testTorch).flatten()
    #acc  = TP+TN (#correctPred)/ TP+TN+FP+FN (all datapoints in )         
    acc_mean[s-1]=bo_mean.sum().float()/y_testTorch.shape[0]
    acc_sample[s-1]=bo_sample.sum().float()/y_testTorch.shape[0]

    #LOSS
    pred_now=pred_list[s]
    #calculate loss for current s
    loss[s-1]=criterion(pred_now, y_testTorch)

#AUC AND LOSS PLOT
fig, axs=plt.subplots(2, 1, figsize=(9,9))
fig.suptitle('NumOfSamples: ' +str(num_samples) + '/StepsPerSample: ' + str(num_steps_per_sample) + '/Stepsize: ' + str(step_size) + '\n BurnIn: ' + str(burn_in) + '/tau_I: ' +str(tau))
lim=num_samples-burn_in_after
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
plt.savefig('/home/rosa/git/SparseChem/examples/chembl/hamiltorch/HMC_plots/SparseChem_classification/Target1482/AUC_LOSS_plot_OnTrainingSet_SparseChem_stepSize'+str(step_size)+'_numSteps'+str(num_steps_per_sample)+'_burnIn'+str(burn_in_after)+'_numSamples'+str(num_samples)+'_hiddenSize'+str(args.hidden_sizes)+'_numTraining'+str(x_train.shape[0])+'_tauOut'+str(round(tau_out, 2))+'_tauList'+str(tau)+'.png')'''


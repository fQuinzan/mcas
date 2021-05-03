import models

import algos as alg
import math as mt
import numpy as np
import pandas as pd
import time as tm
import sys
import random

def run_experiment(model, k_range, eps_FAST_OMP, tau,N_samples, SDS_OMP = True, FAST_OMP = True, SDS_MA = True, Top_k = True) :

    '''
    Run a set of experiments for selected algorithms. All results are saved to text files.
    
    INPUTS:
    
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k_range -- range for the solution size to test experiments with
    
    eps_FAST_OMP -- parameter epsilon for FAST_OMP
    tau -- parameter m/M for the FAST_OMP
    
    N_samples -- number of runs for the randomized algorithms
    SDS_OMP -- if True the SDS_OMP algorithm is tested
    FAST_OMP -- if True the FAST_OMP algorithm is tested
    SDS_MA -- if True the SDS_MA algorithm is tested
    Top_k -- if True the Top_k algorithm is tested
    '''


        
    # ----- run FAST_OMP
    if SDS_OMP :
        print('----- testing SDS_OMP')
        alg.SDS_OMP(model, k_range[-1])


        
    # ----- run Top_k
    if Top_k :
        print('----- testing Top_k')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'rounds_ind': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = alg.Top_k(k_range[j], model)
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k'] = k_range[j]
            results.loc[j,'time']   = out[0]
            results.loc[j,'rounds'] = out[1]
            results.loc[j,'rounds_ind'] = out[2]
            results.loc[j,'metric'] = out[3]
            results.to_csv('Top_k.csv', index = False)

        
        
    # ----- run FAST_OMP
    if FAST_OMP :
        print('----- testing FAST_OMP')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time_mn': np.zeros(len(k_range)), 'rounds_mn': np.zeros(len(k_range)),'metric_mn': np.zeros(len(k_range)), 'time_sd': np.zeros(len(k_range)), 'rounds_sd': np.zeros(len(k_range)),'metric_sd': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = [alg.FAST_OMP(model, k_range[j], eps_FAST_OMP, tau) for i in range(N_samples)]
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k']         = k_range[j]
            results.loc[j,'time_mn']   = np.mean([out[i,0] for i in range(N_samples)])
            results.loc[j,'rounds_mn'] = np.mean([out[i,1] for i in range(N_samples)])
            results.loc[j,'rounds_ind_mn'] = np.mean([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_mn'] = np.mean([out[i,3] for i in range(N_samples)])
            results.loc[j,'time_sd']   = np.std([out[i,0]  for i in range(N_samples)])
            results.loc[j,'rounds_sd'] = np.std([out[i,1]  for i in range(N_samples)])
            results.loc[j,'rounds_ind_sd'] = np.std([out[i,2] for i in range(N_samples)])
            results.loc[j,'metric_sd'] = np.std([out[i,3]  for i in range(N_samples)])
            results.to_csv('FAST_OMP.csv', index = False)



    # ----- run SDS_MA
    if SDS_MA :
        print('----- testing SDS_MA')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        alg.SDS_MA(model, k_range[-1])
    
    return
    
    
'''
Test algorithms with the run_experiment function.
target -- the observations for the regression
features -- the feature matrix for the regression
model -- choose if 'logistic' or 'linear' regression
k_range -- range for the parameter k for a set of experiments
SDS_OMP -- if True, test this algorithm
FAST_OMP -- if True, test this algorithm
SDS_MA -- if True, test this algorithm
DASH -- if True, test this algorithm
Top_k -- if True, test this algorithm
eps_FAST_OMP -- parameter epsilon for FAST_OMP
eps_DASH -- parameter epsilon for DASH
tau -- parameter m/M for the FAST_OMP
alpha -- parameter (m/M)^2 for the FAST_OMP
r -- number of outer iterations for DASH
N_samples -- number of runs for the randomized algorithms
This set of experiment was tested on Python 3.6.5, with MacBook Pro with processor 2,7 GHz Dual-Core Intel Core i5 and 8 GB 1867 MHz DDR3 memory. The parameters for all algorithms may not be optimally tuned. The performance of each algorithm is affected by the machine used to run the algorithms.
'''



def multivariate(mean, Sigma, samples) :
    
    
    L = np.linalg.cholesky(Sigma)
    
    for i in range(samples) :
        
        mean = np.random.normal(loc=0.0, scale=1.0, size=L.shape[0])
        mean = np.dot(L, mean.transpose()) + mean
        write = pd.DataFrame(mean).T

        if i == 0 : write.to_csv('features.csv', index = False, header = False, mode = 'w')
        else : write.to_csv('features.csv', index = False, header = False, mode = 'a')
    
    return



# define features and target for the experiments

n_features = 200
n_samples  = 10

# create features

mean = np.zeros(n_features)
Sigma = np.identity(n_samples)

df = multivariate(mean, Sigma, n_samples)
df = pd.read_csv('features.csv', index_col=0, parse_dates=False)
df = pd.DataFrame(df)

features = df

# create features

arr_ones = np.array([1]*mt.floor(features.shape[1]/10) + [0]*(features.shape[1] - mt.floor(features.shape[1]/10))).astype(float)
np.random.shuffle(arr_ones)
for i in range(features.shape[1]) : arr_ones[i] = arr_ones[i] * random.uniform(-2, 2)
target = np.array([])
for i in range(features.shape[0]):
    arr_two = arr_ones + np.random.rand(arr_ones.shape[0],) * 0.001
    target = np.append(target, np.dot(np.array(features.iloc[i, :]), arr_two))
target = pd.DataFrame(target)
target.to_csv('target.csv', index = False, header = False, mode = 'w')

# saver results

target = df.iloc[:,0]
features = df.iloc[:, range(1, df.shape[1] )]
del(df)

models.init_worker(features, target)
del(features)
del(target)

# choose if logistic or linear regression
model = 'linear'

# set range for the experiments
k_range = np.array([2,3, 5, 6])

# choose algorithms to be tested
SDS_OMP  = True
FAST_OMP = True
Top_k    = True
SDS_MA   = True

# define parameters for algorithms
eps = 0.01
tau = 0.00001
# number of samples per evaluation
N_samples = 1

# run experiment
run_experiment(model = model, k_range = k_range, eps_FAST_OMP = eps, tau = tau, N_samples = N_samples, SDS_OMP = SDS_OMP, SDS_MA = SDS_MA, FAST_OMP = FAST_OMP, Top_k = Top_k)

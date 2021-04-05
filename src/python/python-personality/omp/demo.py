import models

import algos as alg
import math as mt
import numpy as np
import pandas as pd
import time as tm
from sys import getsizeof

import cProfile, pstats, io

def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner
#@profile
def run_experiment(features, target, model, k_range, SDS_OMP = True, SDS_MA = True, Top_k = True) :

    '''
    Run a set of experiments for selected algorithms. All results are saved to text files.
    
    INPUTS:
    
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k_range -- range for the solution size to test experiments with

    SDS_OMP -- if True the SDS_OMP algorithm is tested
    SDS_MA -- if True the SDS_MA algorithm is tested
    Top_k -- if True the Top_k algorithm is tested

    '''
    # ----- run SDS_OMP
    if SDS_OMP :
        print('----- testing SDS_OMP')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
             
            # perform experiments
            out = alg.SDS_OMP(features, target, model, k_range[j])
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k'] = k_range[j]
            results.loc[j,'time']   = out[0]
            results.loc[j,'rounds'] = out[1]
            results.loc[j,'metric'] = out[2]
            results.to_csv('SDS_OMP.csv', index = False)
        #res = alg.SDS_OMP(features, target, model, k_range[-1])
        
        
        
    # ----- run SDS_MA
    if SDS_MA :
        print('----- testing SDS_MA')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = alg.SDS_MA(features, target, model, k_range[j])
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k'] = k_range[j]
            results.loc[j,'time']   = out[0]
            results.loc[j,'rounds'] = out[1]
            results.loc[j,'metric'] = out[2]
            results.to_csv('SDS_MA.csv', index = False)
        
        
        
    # ----- run Top_k
    if Top_k :
        print('----- testing Top_k')
        results = pd.DataFrame(data = {'k': np.zeros(len(k_range)).astype('int'), 'time': np.zeros(len(k_range)), 'rounds': np.zeros(len(k_range)),'metric': np.zeros(len(k_range))})
        for j in range(len(k_range)) :
        
            # perform experiments
            out = alg.Top_k(features, target, k_range[j], model)
            out = np.array(out)
            
            # save data to file
            results.loc[j,'k'] = k_range[j]
            results.loc[j,'time']   = out[0]
            results.loc[j,'rounds'] = out[1]
            results.loc[j,'metric'] = out[2]
            results.to_csv('Top_k.csv', index = False)
    
    return
    
    
'''

Test algorithms with the run_experiment function.

target -- the observations for the regression
features -- the feature matrix for the regression
model -- choose if 'logistic' or 'linear' regression

k_range -- range for the parameter k for a set of experiments

SDS_OMP -- if True, test this algorithm
SDS_MA -- if True, test this algorithm
Top_k -- if True, test this algorithm

'''

# define features and target for the experiments
df = pd.read_csv('data.csv', index_col=0, parse_dates=False)
df = pd.DataFrame(df)
target = df.iloc[:, -1]
features = df.iloc[:, range(df.shape[1] - 1)]
# choose if logistic or linear regression
model = 'logistic'

# set range for the experiments
k_range = range(3, 4)

# choose algorithms to be tested
SDS_OMP  = True 
Top_k    = False 
SDS_MA   = False
# run experiment
run_experiment(features, target, model = model, k_range = k_range, SDS_OMP = SDS_OMP, SDS_MA = SDS_MA, Top_k = Top_k)

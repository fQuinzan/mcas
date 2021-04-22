import functools
import math
import models
import random
import sys
import time

import multiprocessing as mp
import numpy as np
import pandas as pd

from joblib import Parallel, delayed, parallel_backend



# ------------------------------------------------------------------------------------------
#  additional functions
# ------------------------------------------------------------------------------------------

var_dict = {}

def init_worker(features, target):

    features = np.array(features)
    features_arr = mp.RawArray('d', features.shape[0]*features.shape[1])
    features_np = np.frombuffer(features_arr, dtype = np.dtype(float)).reshape(features.shape)
    np.copyto(features_np, features)
    
    target = np.array(target)
    target_arr = mp.RawArray('d', target.shape[0])
    target_np = np.frombuffer(target_arr, dtype = np.dtype(float)).reshape(target.shape)
    np.copyto(target_np, target)

    var_dict['features']       = features_np
    var_dict['features_shape'] = features.shape
    var_dict['target']         = target_np
    var_dict['target_shape']   = target.shape
 
 
 
def argmax_with_condition(arr, idx) :

    arr = np.array(arr)
    idx = np.array(idx)

    res = 0
    val = arr[0]
    
    for i in np.setdiff1d(range(len(arr)), idx) :
        if arr[i] >= val :
            res = i
            val = arr[i]
            
    return res
            


# ------------------------------------------------------------------------------------------
#  The FAST_OMP algorithm
# ------------------------------------------------------------------------------------------
            
        
        
def FAST_OMP_parallel_oracle(i, n_cpus, S, A, X_size, t, model) :

    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])
    
    # define time for oracle calls
    rounds_ind = 0
        
    while i < len(A) :

        # compute gradiet
        out = np.array(models.oracle(features_np, target_np, np.append(S, A[0:i]), model, 'FAST_OMP'), dtype='object')
        vals = np.array([])
        
        # evaluate feasibility and compute Xj
        for j in np.setdiff1d(range(len(out[0])), np.append(S, A[0:i])) :
            rounds_ind += 1
            if models.constraint(features_np, target_np, np.append(S, A[0:i]), j, model, 'FAST_OMP') and out[0][j] >= pow(t, 0.5):
                vals = np.append(vals, j)
        Xj = [vals, out[1], i, False, rounds_ind]
        
        # return points if they fulfill cardinality condition
        if  Xj[0].size < X_size :
            Xj[-2] = True
            return np.array(Xj, dtype='object')
            
        # otherwise theturn the entire sequence
        elif i + n_cpus >= len(A) :
            return np.array(Xj, dtype='object')
        
        # update t
        i = i + n_cpus



def FAST_OMP(features, target, model, k, eps, tau) :

    '''
    The FAST_OMP algorithm, as in Algorithm 1 in the submission
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    eps -- parameter epsilon for the approximation
    tau -- parameter m/M for the (M,m)-(restricted smootheness, restricted strong concavity)
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    '''

    # define time and rounds
    run_time = time.time()
    rounds = 0
    rounds_ind = 0

    # define initial solution and number of outer iterations
    S = np.array([], int)
    iter = 0
    
    # copy features and target to shared array
    grad, metric = models.oracle(features, target, S, model, algo = 'FAST_OMP')
    features = features.iloc[:,np.array(np.where(grad >= 0)[0])]
    init_worker(features, target)
    rounds += 1
    
    # redefine k
    k = min(k, features.shape[1] - 1)
    
    # multiprocessing
    N_CPU = mp.cpu_count()
    pool = mp.Pool(N_CPU)
    
    while iter <= 1/eps and S.size < k :

        # define new set X and copy to array
        X = np.setdiff1d(np.array(range(features.shape[1]), int), S)
        
        # find largest k-elements and update adaptivity
        if S.size != 0 :
            grad, metric = models.oracle(features, target, S, model, algo = 'FAST_OMP')
            rounds += 1
        
        # define parameter t
        t = np.power(grad, 2)
        t = np.sort(t)[::-1]
        t = (1 - eps) * tau *  np.sum(t[range(k)]) / k
        
        print(t)

        while X.size > 0 and S.size < k :

            # define new random set of features and random index
            A = np.random.choice(np.setdiff1d(X, S), min(k - S.size, np.setdiff1d(X, S).size), replace=False)

            # compute the increments in parallel
            X_size = (1 - eps) * X.size
            N_PROCESSES = min(N_CPU, len(A))
            out = pool.map(functools.partial(FAST_OMP_parallel_oracle, n_cpus = N_PROCESSES, S = S, A = A, X_size = X_size, t = t, model = model), range(N_PROCESSES))
            out = np.array(out)
            
            # manually update rounds
            rounds += math.ceil(len(A)/N_PROCESSES)
            rounds_ind += max(out[:, -1])
                
            # compute sets Xj for the good incement
            idx = np.argsort(out[:, -3])
            for j in idx :
                if  out[j, -2] == True or j == idx[-1]:
                    S = np.append(S, A[0:(out[j, -3] + 1)])
                    S = np.unique(S[S >= 0])
                    X = out[j, 0]
                    metric = out[j, 1]
                    break

        iter = iter + 1;
        
    # update current time
    run_time = time.time() - run_time
    pool.close()
    pool.join()

    return run_time, rounds, rounds_ind, metric



# ------------------------------------------------------------------------------------------
#  The SDS_OMP algorithm
# ------------------------------------------------------------------------------------------



def SDS_OMP_parallel_oracle(A, S, model) :

    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])
    
    # define vals
    point = []
    constraint = []
    for a in np.setdiff1d(A, S) :
        if models.constraint(features_np, target_np, S, a, model, 'SDS_OMP') :
            point = np.append(point, a)
        
    return point, len(np.setdiff1d(A, S))



def SDS_OMP(features, target, model, k) :

    '''
    The SDS_OMP algorithm, as in "Submodular Dictionary Selection for Sparse Representation", Krause and Cevher, ICML '10
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    '''

    # multiprocessing
    init_worker(features, target)
    N_CPU = mp.cpu_count()
    pool = mp.Pool(N_CPU)
    
    # save data to file
    results = pd.DataFrame(data = {'k': np.zeros(k).astype('int'), 'time': np.zeros(k), 'rounds': np.zeros(k),'metric': np.zeros(k)})

    # define time and rounds
    run_time = time.time()
    rounds = 0
    rounds_ind = 0

    # define new solution
    S = np.array([], int)
    
    for idx in range(k) :
    
        # define and train model
        grad, metric = models.oracle(features, target, S, model, algo = 'SDS_OMP')
        rounds += 1
        
        # evaluate feasibility in parallel
        N_PROCESSES = min(N_CPU, len(grad))
        out = pool.map(functools.partial(SDS_OMP_parallel_oracle, S = S, model = model), np.array_split(range(len(grad)), N_PROCESSES))
        out = np.array(out, dtype='object')
        rounds_ind += np.max(out[:, -1])
        
        # save results to file
        results.loc[idx,'k']      = idx + 1
        results.loc[idx,'time']   = time.time() - run_time
        results.loc[idx,'rounds'] = int(rounds)
        results.loc[idx,'rounds_ind'] = rounds_ind
        results.loc[idx,'metric'] = metric
        results.to_csv('SDS_OMP.csv', index = False)
        
        # get feasible points
        points = np.array([])
        for i in range(N_PROCESSES) : points = np.append(points, np.array(out[i, 0]))
        points = points.astype('int')
        
        # break if points are no longer feasible
        if len(points) == 0 : break
        
        # otherwise add maximum point to current solution
        a = points[0]
        for i in points :
            if grad[i] > grad[a] :
                a = i
                
        if grad[a] >= 0 :
            S  = np.unique(np.append(S,i))
        else : break
        
    # update current time
    run_time = time.time() - run_time
    pool.close()
    pool.join()

    return run_time, rounds, metric
    
    
    
# ------------------------------------------------------------------------------------------
#  The SDS_MA algorithm
# ------------------------------------------------------------------------------------------



def SDS_MA_parallel_oracle(A, S, fS, model) :

    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])
    
    # define vals
    marginal = 0
    res = [0, 0, 0]

    for a in A :
        out = models.oracle(features_np, target_np, np.append(S, a), model, 'SDS_MA')
        idx = models.constraint(features_np, target_np, S, a, model, 'SDS_MA')
        if out - fS >= marginal and idx :
            res = [a, out, idx]
            marginal = out
             
    return res



def SDS_MA(features, target, model, k) :

    '''
    The SDS_MA algorithm, as in "Submodular Dictionary Selection for Sparse Representation", Krause and Cevher, ICML '10
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    '''
    
    # save data to file
    results = pd.DataFrame(data = {'k': np.zeros(k).astype('int'), 'time': np.zeros(k), 'rounds': np.zeros(k),'metric': np.zeros(k)})

    # measure time and adaptivity
    run_time = time.time()
    rounds = 1
    rounds_ind = 0
    
    # define initial solution, ground set, and metric
    S = np.array([], int)
    X = np.array(range(features.shape[1]))
    metric = models.oracle(features, target, S, model, 'SDS_MA')
    
    # multiprocessing
    init_worker(features, target)
    N_CPU = mp.cpu_count()
    pool = mp.Pool(N_CPU)
    for idx in range(k) :
    
        # define points that need to be added
        if X.shape == 0 : break

        # evaluate points in parallel
        N_PROCESSES = min(N_CPU, len(X))
        out = pool.map(functools.partial(SDS_MA_parallel_oracle, S = S, fS = metric, model = model), np.array_split(X, N_PROCESSES))
        out = np.array(out)
        
        # manually update rounds
        rounds += math.ceil(len(X)/N_PROCESSES)
        rounds_ind += math.ceil(len(X)/N_PROCESSES)
        
        # save results to file
        results.loc[idx,'k']      = idx + 1
        results.loc[idx,'time']   = time.time() - run_time
        results.loc[idx,'rounds'] = int(rounds)
        results.loc[idx,'rounds_ind'] = int(rounds_ind)
        results.loc[idx,'metric'] = metric
        results.to_csv('SDS_MA.csv', index = False)

        # break if the solutions are not feasible
        if len(np.where(out[:, -1] == True)[0]) == 0 : break
        
        # otherwise find best solution and add it to set
        i = argmax_with_condition(out[:, 1], np.where(out[:, -1] == False)[0])
        
        # if the increment holds
        if out[i, -2] - metric >= 0 :
            metric = out[i, -2]
            S = np.append(S, int(out[i, 0]))
            S = np.unique(S[S >= 0])
            X = np.setdiff1d(X, int(out[i, 0]))
            
        # otherwise break
        else : break
        
    # evaluate solution quality (????)
    #if model == 'logistic' : metric = -metric
    
    # update current time
    run_time = time.time() - run_time
    pool.close()
    pool.join()
    
    return run_time, rounds, rounds_ind, metric
    
    
    
# ------------------------------------------------------------------------------------------
#  The Top_k algorithm
# ------------------------------------------------------------------------------------------

def TOP_K_parallel_oracle(X, model) :

    # get variables from shared array
    features_np = np.frombuffer(var_dict['features'], dtype = np.dtype(float)).reshape(var_dict['features_shape'])
    target_np   = np.frombuffer(var_dict['target'], dtype = np.dtype(float)).reshape(var_dict['target_shape'])
    
    # define vals
    res = pd.DataFrame(data = {'point': np.zeros(len(X)).astype('int'), 'metric': np.zeros(len(X))})

    idx = 0
    for a in X :
        out = models.oracle(features_np, target_np, a, model, 'SDS_MA')
        res.loc[idx,'point']  = int(a)
        res.loc[idx,'metric'] = out
        idx += 1
        
    return res

def Top_k(features, target, k, model) :

    '''
    This algorithm selects the k features with the highest score
    
    INPUTS:
    features -- the feature matrix
    target -- the observations
    model -- choose if the regression is linear or logistic
    k -- upper-bound on the solution size
    OUTPUTS:
    float run_time -- the processing time to optimize the function
    int rounds -- the number of parallel calls to the oracle function
    float metric -- a goodness of fit metric for the solution quality
    '''

    # measure time and adaptivity
    run_time = time.time()
    rounds = 0
    rounds_ind = 0
    
    # define ground set and initial solution
    S = np.array([], int)
    T = np.array([], int)
    X = np.array(range(features.shape[1]))
    
    # multiprocessing
    init_worker(features, target)
    N_CPU = mp.cpu_count()
    pool = mp.Pool(N_CPU)
    
    # find marginal contributions
    marginal = np.array([])
    
    # evaluate points in parallel
    N_PROCESSES = min(N_CPU, len(X))
    out = pool.map(functools.partial(TOP_K_parallel_oracle, model = model), np.array_split(X, N_PROCESSES))
    
    # manually update rounds
    rounds += math.ceil(len(X)/N_PROCESSES)

    # postrpcess results
    point = np.array([])
    metric = np.array([])
    for i in range(N_PROCESSES) :
        point = np.append(point, np.array(out[i]['point']))
        metric = np.append(metric, np.array(out[i]['metric']))
    
    # add top k points to current solution
    while len(S) < k :
    
        # find maximum element that was not already added
        max = argmax_with_condition(metric, T)
        if models.constraint(features, target, S, int(point[max]), model, 'Top_k') :
            S = np.append(S, int(point[max]))
        T = np.append(T, int(point[max]))
        rounds_ind += 1
        
        if metric.size == len(T) : break
        
    # update current time
    run_time = time.time() - run_time
    
    # evaluate solution quality
    metric = models.oracle(features, target, S, model, 'Top_k')
    pool.close()
    pool.join()

    return run_time, rounds, rounds_ind, metric

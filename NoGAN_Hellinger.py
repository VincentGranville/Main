import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot

# you can read data from URL below
# https://raw.githubusercontent.com/VincentGranville/Main/main/circle8d.csv
data = pd.read_csv('circle8d.csv')


#--- [1] read data, build quantile structure 

X = data.to_numpy()
nobs_real, dim = data.shape
# granularity: ideal between 40 and 400 (needs n_iter > 10000)
granularity = 60 
Hyperparam = np.full(dim, granularity) 
arr_q = []

for d in range(dim): 
    n = Hyperparam[d]
    arr_qd = np.zeros(n+1)
    for k in range(n):
        arr_qd[k] = np.quantile(X[:,d], k/n)
    arr_qd[n] = max(X[:,d])
    arr_q.append(arr_qd)


#--- [2] Build bin structure for real data

def find_quantile_index(x, arr_quantiles):
    k = 0 
    while x > arr_quantiles[k] and k < len(arr_quantiles): 
        k += 1
    return(max(0, k-1)) 


def create_bin_structure(x, arr_q):

    hash_bins = {}
    hash_bins_median = {}
    hash_index = []

    for n in range(x.shape[0]):

        key = ()
        for d in range(dim):
            kd = find_quantile_index(x[n,d], arr_q[d])
            key = (*key, kd)        
        hash_index.append(key)

        if key in hash_bins: 
            hash_bins[key] += 1
            points = hash_bins_median[key]
            points.append(x[n,:])
            hash_bins_median[key] = points 
        else:
            hash_bins[key] = 1
            hash_bins_median[key] = [x[n,:]]  

    for key in hash_bins: 
        points = hash_bins_median[key]
        # beware: even number of points -> median is not one of the points  
        median = np.median(points, axis = 0) 
        hash_bins_median[key] = median

    return(hash_bins, hash_index, hash_bins_median)


( hash_bins_real, 
  hash_index_real, 
  hash_bins_median_real,
) = create_bin_structure(X, arr_q)


#--- [3] Generate nobs_synth obs, create initial bin structure for synth data  

# if nobs_synth > 1000, split in smaller batches, do one batch at a time
nobs_synth = nobs_real 
seed = 155
np.random.seed(seed)

synth_X = []
for d in range(dim):
    synth_xd = np.random.uniform(min(X[:,d]), max(X[:,d]), nobs_synth)
    synth_X.append(synth_xd)
synth_X = np.transpose(np.array(synth_X))

( hash_bins_synth, 
  hash_index_synth,
  hash_bins_median_synth,  # unused
) = create_bin_structure(synth_X, arr_q)


#--- [4] Main part: change synth obs to minimize Hellinger loss function

def in_bin(x, key, arr_q):
    # test if vector x is in bin attached to key
    status = True
    for d in range(dim):
        arr_qd = arr_q[d]
        kd = key[d]
        if x[d] < arr_qd[kd] or x[d] >= arr_qd[kd+1]:
            status = False  # x is not in the bin
    return(status) 


n_iter = 10000 
Hellinger = 2.0  # maximum potential value (min is 0 and means perfect fit)

for iter in range(n_iter):

    # random point k with bin_k in synth data to be replaced with point x sampled 
    #   in bin_l, with bin_l attached to point l randomly chosen in real data

    l = np.random.randint(0, nobs_real) 
    key_l = hash_index_real[l]
    if key_l in hash_bins_synth:
        scount2 = hash_bins_synth[key_l]
    else:
        scount2 = 0 
    rcount2 = hash_bins_real[key_l] 

    tries = 1
    k = np.random.randint(0, nobs_synth) 
    key_k = hash_index_synth[k]
    while key_k in hash_bins_real and tries < 10:
        # loop on k until key_k not in hash_bins_real
        k = np.random.randint(0, nobs_synth)  
        key_k = hash_index_synth[k]
        tries += 1
    scount1 = hash_bins_synth[key_k] 
    if key_k in hash_bins_real:
        rcount1 = hash_bins_real[key_k]
    else:
        rcount1 = 0

    # compute change in Helliger distance with proposed update
    A = - ( np.sqrt(scount1/nobs_synth) - np.sqrt(rcount1/nobs_real) )**2
    B = + ( np.sqrt((scount1-1)/nobs_synth) - np.sqrt(rcount1/nobs_real) )**2
    C = - ( np.sqrt(scount2/nobs_synth) - np.sqrt(rcount2/nobs_real) )**2
    D = + ( np.sqrt((scount2+1)/nobs_synth) - np.sqrt(rcount2/nobs_real) )**2
    delta_H = A + B + C + D

    # if delta_H < 0.00:

    if (delta_H < 0.00 and Hellinger < 0.6)  or Hellinger + delta_H > 0.6: 

        # assign point k in synth data to bin attached to point l in real data

        Hellinger += delta_H
        hash_index_synth[k] = key_l
        if hash_bins_synth[key_k] == 1:
            del hash_bins_synth[key_k]
        else: 
            hash_bins_synth[key_k] -= 1
        if key_l in hash_bins_synth:
            hash_bins_synth[key_l] += 1
        else:
            hash_bins_synth[key_l] = 1

        # replace k-th synth obs synth_X[k,:] by vector x sampled in bin key_l in real data
        
        sampling_mode = 'Gaussian'   # options: 'Median', 'Uniform', 'Gaussian'

        if sampling_mode == 'Gaussian':  

            scale = 1.00   # the lower, the more faithful the synth data
            cov = []
            for d in range(dim):
                ld = key_l[d]
                arr_qd = arr_q[d]
                cov.append((arr_qd[ld+1] - arr_qd[ld])**2)
            cov = scale * np.diag(cov)
            median = hash_bins_median_real[key_l]
        
            x = np.random.multivariate_normal(median,cov)  
            tries = 1
            # the following (truncated Gaussian) is to keep x inside bin_l
            while tries < 5 and not in_bin(x, key_l, arr_q):  
                x = np.random.multivariate_normal(median,cov)
                tries += 1
            synth_X[k, :] = x

        elif sampling_mode == 'Uniform':

            for d in range(dim):
                ld = key_l[d]
                arr_qd = arr_q[d]
                synth_X[k,d] = np.random.uniform(arr_qd[ld], arr_qd[ld+1])

        elif sampling_mode == 'Median':

            # ideal for categorical features [force this mode for these features]
            synth_X[k,:] = hash_bins_median_real[key_l]

    print("Hellinger dist, synth vs real: %8.5f" %(Hellinger)) 


#--- [5] Plot some result

mpl.rcParams['lines.linewidth'] = 0.3
mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

plt.scatter(synth_X[:,0],synth_X[:,1], c = 'red', s = 0.6)
plt.scatter(X[:,0],X[:,1], c = 'blue', s = 0.6)
plt.show()

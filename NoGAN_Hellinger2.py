import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot


#--- [1] Read data and select dim 

# you can read data from URL below
# https://raw.githubusercontent.com/VincentGranville/Main/main/circle8d.csv
data = pd.read_csv('circle8d.csv')
features = list(data.columns.values)
X = data.to_numpy()
features = np.array(features)

# use the first dim columns only
dim = 2 
X = X[:, 0:dim] 
features = features[0:dim]
nobs_real, dim = X.shape


#--- [2] Functions to build bin structure

def create_quantile_table(x, Hyperparam, shift):

    arr_q = [] 
    for d in range(dim): 
        n = Hyperparam[d]
        arr_qd = np.zeros(n+1)
        for k in range(n):
            q = shift[d] + k/n
            arr_qd[k] = np.quantile(x[:,d], q % 1)
        arr_qd[n] = max(x[:,d])
        arr_qd.sort()
        arr_q.append(arr_qd)
    return(arr_q)


def find_quantile_index(x, arr_quantiles):
    k = 0
    while k < len(arr_quantiles) and x > arr_quantiles[k]: 
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


#--- [3] Generate nobs_synth obs

# if nobs_synth > 1000, split in smaller batches, do one batch at a time
nobs_synth = nobs_real  
seed = 155
np.random.seed(seed)

# get initial synth. data with same marginal distributions as real data 

mode = 'Shuffle'   # options: 'Shuffle', 'Quantiles' 
synth_X = np.empty(shape=(nobs_synth,dim))

if mode == 'Quantiles':
    for k in range(nobs_synth):
        pc = np.random.uniform(0, 1.00000001, dim)
        for d in range(dim):
            synth_X[k, d] = np.quantile(X[:,d], pc[d], axis=0)

elif mode == 'Shuffle':
    nobs_synth == nobs_real  # both must be equal
    synth_X = np.copy(X)
    for d in range(dim):
        col = synth_X[:, d]
        np.random.shuffle(col)
        synth_X[:, d] = col

synth_X_init = np.copy(synth_X)


#--- [4] Main part: create synth obs to minimize Hellinger loss function

def in_bin(x, key, arr_q):
    # test if vector x is in bin attached to key
    status = True
    for d in range(dim):
        arr_qd = arr_q[d]
        kd = key[d]
        if x[d] < arr_qd[kd] or x[d] >= arr_qd[kd+1]:
            status = False  # x is not in the bin
    return(status) 

def array_to_tuple(arr):
    list = ()
    for k in range(len(arr)):
        list = (*list, arr[k])
    return(list)

Hellinger = 40.0   # arbitrary value
swaps = 0  
history_log_H = []
history_log_swaps = []
flist = []  # list of image filenames for the video
frame = 0   # frame number, for video

# to accelerate computations (pre-computed sqrt)
sqrt_real = np.sqrt(nobs_real)
sqrt_synth = np.sqrt(nobs_synth)
n_sqrt = max(nobs_real, nobs_synth) 
arr_sqrt = np.sqrt(np.arange(n_sqrt))

# visualization: graphic parameters
mpl.rcParams['lines.linewidth'] = 0.3
mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

video_mode = False

# Hyperparameters
reset_granularity = 10000  # set to 50000 if dim = 3, set to 10000 if dim = 2    
reset_shift = 999999999999 
granularity = 20   #  set to 2 if dim > 2, set to 20 if dim = 2 
Hyperparam = np.full(dim, granularity) 
shift = np.zeros(dim)
n_iter = 2000000  


for iter in range(n_iter):

    if iter % reset_granularity == 0 or iter % reset_shift == 0 or iter == 0:   

        # Get more granular Hellinger approximation

        if iter % reset_granularity == 0: 
            Hyperparam = 1 + Hyperparam    
        if iter % reset_shift == 0: 
            shift = np.random.uniform(0, 1, dim) 
        arr_q = create_quantile_table(X, Hyperparam, shift) 
        ( hash_bins_real, 
          hash_index_real, 
          hash_bins_median_real 
        ) = create_bin_structure(X, arr_q)
        ( hash_bins_synth, 
          hash_index_synth,
          hash_bins_median_synth,  # unused
        ) = create_bin_structure(synth_X, arr_q)

    k = np.random.randint(0, nobs_synth) 
    key_k = hash_index_synth[k]
    scount1 = hash_bins_synth[key_k]
    if key_k in hash_bins_real:
        rcount1 = hash_bins_real[key_k]
    else:
        rcount1 = 0

    l = np.random.randint(0, nobs_synth)  
    key_l = hash_index_synth[l]
    scount2 = hash_bins_synth[key_l]
    if key_l in hash_bins_real:
        rcount2 = hash_bins_real[key_l]
    else:
        rcount2 = 0

    d = np.random.randint(1,dim)  # column 0 can stay fixed

    new_key_k = np.copy(key_k)
    new_key_l = np.copy(key_l)
    new_key_k[d] = key_l[d]
    new_key_l[d] = key_k[d]
    new_key_k = array_to_tuple(new_key_k)
    new_key_l = array_to_tuple(new_key_l)

    if new_key_k in hash_bins_synth:
        scount3 = hash_bins_synth[new_key_k]
    else:
        scount3 = 0
    if new_key_k in hash_bins_real:
        rcount3 = hash_bins_real[new_key_k]
    else:
        rcount3 = 0

    if new_key_l in hash_bins_synth:
        scount4 = hash_bins_synth[new_key_l]
    else:
        scount4 = 0
    if new_key_l in hash_bins_real:
        rcount4 = hash_bins_real[new_key_l]
    else:
        rcount4 = 0

    A = arr_sqrt[scount1]  /sqrt_synth - arr_sqrt[rcount1]/sqrt_real
    B = arr_sqrt[scount1-1]/sqrt_synth - arr_sqrt[rcount1]/sqrt_real
    C = arr_sqrt[scount2]  /sqrt_synth - arr_sqrt[rcount2]/sqrt_real
    D = arr_sqrt[scount2-1]/sqrt_synth - arr_sqrt[rcount2]/sqrt_real
    E = arr_sqrt[scount3]  /sqrt_synth - arr_sqrt[rcount3]/sqrt_real
    F = arr_sqrt[scount3+1]/sqrt_synth - arr_sqrt[rcount3]/sqrt_real
    G = arr_sqrt[scount4]  /sqrt_synth - arr_sqrt[rcount4]/sqrt_real
    H = arr_sqrt[scount4+1]/sqrt_synth - arr_sqrt[rcount4]/sqrt_real
    delta_H = - A*A + B*B - C*C + D*D - E*E + F*F - G*G + H*H

    if delta_H < -0.00001:

        Hellinger += delta_H
        swaps += 1

        # update hash_index_synth and hash_bins_synth

        hash_index_synth[k] = new_key_k
        if new_key_k in hash_bins_synth:
            hash_bins_synth[new_key_k] +=1
        else:
            hash_bins_synth[new_key_k] = 1
        if hash_bins_synth[key_k] == 1:
            del hash_bins_synth[key_k]
        else: 
            hash_bins_synth[key_k] -= 1
   
        hash_index_synth[l] = new_key_l
        if new_key_l in hash_bins_synth:
            hash_bins_synth[new_key_l] += 1
        else:
            hash_bins_synth[new_key_l] =1
        if key_l in hash_bins_synth:
            hash_bins_synth[key_l] -= 1
        else:
            hash_bins_synth[key_l] = 1

        # update synthetic data

        aux = synth_X[k, d]
        synth_X[k, d] = synth_X[l, d]
        synth_X[l, d] = aux

        if video_mode and swaps % 25 == 0:

            # save image for future inclusion in video
            fname='nogan3_frame'+str(frame)+'.png'
            flist.append(fname)
            plt.scatter(synth_X[:,0], synth_X[:,1], s = 1.0) 
            plt.savefig(fname, dpi = 200)
            plt.close() 
            frame += 1

    if iter % 1000 == 0:

        print("Iter: %7d | Loss: %9.6f | Swaps: %5d" 
              %(iter, Hellinger, swaps)) 
        history_log_H.append(Hellinger)
        history_log_swaps.append(swaps)


#--- [5] Evaluation with KS distance

import genai_evaluation as ge

n_nodes = 1000

df_init = pd.DataFrame(synth_X_init, columns = features)
df_synth = pd.DataFrame(synth_X, columns = features)
df_train = pd.DataFrame(X, columns = features) 

query_lst, ecdf_train, ecdf_init = ge.multivariate_ecdf(df_train, 
                df_init, n_nodes, verbose = True) 
ks_base = ge.ks_statistic(ecdf_train, ecdf_init)

query_lst, ecdf_train, ecdf_synth = ge.multivariate_ecdf(df_train, 
                df_synth, n_nodes, verbose = True) 
ks = ge.ks_statistic(ecdf_train, ecdf_synth)

query_lst, ecdf_init, ecdf_synth = ge.multivariate_ecdf(df_init, 
                df_synth, n_nodes, verbose = True) 
ks_diff = ge.ks_statistic(ecdf_init, ecdf_synth)

print("Test ECDF Kolmogorof-Smirnov dist. (synth. vs train.): %6.4f" %(ks))
print("Base ECDF Kolmogorof-Smirnov dist. (init.  vs train.): %6.4f" %(ks_base))
print("Diff ECDF Kolmogorof-Smirnov dist. (init.  vs synth.): %6.4f" %(ks_diff))


#--- [6] Plot some results and create video

mpl.rc('hatch', color='k', linewidth=0.3)
plt.scatter(X[:,0],X[:,1],marker='o',c='deepskyblue',alpha=0.1,s=10) 
plt.scatter(synth_X[:,0],synth_X[:,1],marker='o',c='coral',alpha=0.4,s=10, edgecolors='black',lw=0.2)
plt.grid(linewidth = 0.4, alpha = 1)
plt.show()

x_axis = range(len(history_log_H))
plt.plot(x_axis, history_log_H)
plt.show()
plt.plot(x_axis, history_log_swaps)
plt.show()

if video_mode:
    import moviepy.video.io.ImageSequenceClip
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(flist, fps=6)
    clip.write_videofile('nogan3.mp4')


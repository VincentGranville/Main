import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import genai_evaluation as ge
from matplotlib import pyplot
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.simplefilter("ignore")


#--- [1] read data and only keep features and observations we want

#- [1.2] read data

url = "https://raw.githubusercontent.com/VincentGranville/Main/main/diabetes.csv"
# data = pd.read_csv('students_C2_full_nogan.csv')
data = pd.read_csv(url)
print(data)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
            'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

#- [1.2] set seed for replicability

pd.core.common.random_state(None)
seed = 106 ## 105
np.random.seed(seed)

#- [1.3] select features

data = data[features]
data = data.sample(frac = 1)  # shuffle rows to break artificial sorting

#- [1.4] split real dataset into training and validation sets

data_training = data.sample(frac = 0.5)
data_validation = data.drop(data_training.index)
data_training.to_csv('training_vg2.csv')
data_validation.to_csv('validation_vg2.csv')
data_train = pd.DataFrame.to_numpy(data_training) 

nobs = len(data_training)
n_features = len(features)


#--- [2] create initial synthetic data  

def create_initial_synth(nobs_synth):

    eps = 0.000000001
    n_features = len(features)
    data_synth = np.empty(shape=(nobs_synth,n_features))

    for i in range(nobs_synth):
        pc = np.random.uniform(0, 1 + eps, n_features)
        for k in range(n_features):
            label = features[k]
            data_synth[i, k] = np.quantile(data_training[label], pc[k], axis=0)
    return(data_synth)


#--- [3] loss functions Part 1

def compute_univariate_stats():

    # 'dt' for training data, 'ds' for synth. data

    # for first tem in loss function
    dt_mean  = np.mean(data_train, axis=0)
    dt_stdev = np.std(data_train, axis=0)
    ds_mean  = np.mean(data_synth, axis=0)
    ds_stdev = np.std(data_synth, axis=0)

    # for g(arr)
    dt_mean1  = np.mean(g(data_train), axis=0)
    dt_stdev1 = np.std(g(data_train), axis=0)
    ds_mean1  = np.mean(g(data_synth), axis=0)
    ds_stdev1 = np.std(g(data_synth), axis=0)

    # for f(arr)
    dt_mean2  = np.mean(h(data_train), axis=0)
    dt_stdev2 = np.std(h(data_train), axis=0)
    ds_mean2  = np.mean(h(data_synth), axis=0)
    ds_stdev2 = np.std(h(data_synth), axis=0)

    values = [dt_mean, dt_stdev, ds_mean, ds_stdev,
              dt_mean1, dt_stdev1, ds_mean1, ds_stdev1,
              dt_mean2, dt_stdev2, ds_mean2, ds_stdev2]
    return(values)

def initialize_cross_products_tables():

    # the core structure for fast computation when swapping 2 values
    # 'dt' for training data, 'ds' for synth. data
    # 'prod' is for 1st term in loss, 'prod12' for 2nd term

    dt_prod = np.empty(shape=(n_features,n_features))
    ds_prod = np.empty(shape=(n_features,n_features))
    dt_prod12 = np.empty(shape=(n_features,n_features))
    ds_prod12 = np.empty(shape=(n_features,n_features))

    for k in range(n_features):
        for l in range(n_features):
            dt_prod[l, k] = np.dot(data_train[:,l], data_train[:,k])  
            ds_prod[l, k] = np.dot(data_synth[:,l], data_synth[:,k])   
            dt_prod12[l, k] = np.dot(g(data_train[:,l]), h(data_train[:,k])) 
            ds_prod12[l, k] = np.dot(g(data_synth[:,l]), h(data_synth[:,k])) 
    products = [dt_prod, ds_prod, dt_prod12, ds_prod12]
    return(products)
    

#--- [4] loss function Part 2: managing loss function

# Weights hyperparameter:
#
#    1st value is for 1st term in loss function, 2nd value for 2nd term
#    each value should be between 0 and 1, all adding to 1
#    works best when loss contributions from each term are about the same

#- [4.1] loss function contribution from features (k, l) jointly

# before calling functions in sections [4.1], [4.2] and [4.3], first intialize
# by calling compute_univariate_stats() and compute_cross_products() before;
# this initialization needs to be done only once at the beginning

def get_distance(k, l, weights):

    dt_prodn = dt_prod[k, l] / nobs
    ds_prodn = ds_prod[k, l] / nobs_synth
    dt_r = (dt_prodn - dt_mean[k]*dt_mean[l]) / (dt_stdev[k]*dt_stdev[l])
    ds_r = (ds_prodn - ds_mean[k]*ds_mean[l]) / (ds_stdev[k]*ds_stdev[l])

    dt_prodn12 = dt_prod12[k, l] / nobs  
    ds_prodn12 = ds_prod12[k, l] / nobs_synth 
    dt_r12 = (dt_prodn12 - dt_mean1[k]*dt_mean2[l]) / (dt_stdev1[k]*dt_stdev2[l])
    ds_r12 = (ds_prodn12 - ds_mean1[k]*ds_mean2[l]) / (ds_stdev1[k]*ds_stdev2[l])

    # dist = weights[0]*abs(dt_r - ds_r) + weights[1]*abs(dt_r12 - ds_r12) 
    dist = max(weights[0]*abs(dt_r - ds_r), weights[1]*abs(dt_r12 - ds_r12)) 
    return(dist, dt_r, ds_r, dt_r12, ds_r12)
 
def total_distance(weights, flagParam): 

    eval = 0
    max_dist = 0
    super_max = 0
    lmax = n_features

    for k in range(n_features):
        if symmetric:
            lmax = k
        for l in range(lmax):
            if l != k and flagParam[k] > 0 and flagParam[l] >0: 
                values = get_distance(k, l, weights) 
                dist2 = max(abs(values[1] - values[2]), abs(values[3] - values[4]))
                eval += values[0]
                if values[0] > max_dist:
                    max_dist = values[0]
                if dist2 > super_max:
                    super_max = dist2
    return(eval, max_dist, super_max)

#- [4.2] updated loss function when swapping rows idx1 and idx2 in feature k
#        contribution from feature l jointly with k

def get_new_distance(k, l, idx1, idx2, weights):

    tmp1_k = data_synth[idx1, k]
    tmp2_k = data_synth[idx2, k]
    tmp1_l = data_synth[idx1, l]
    tmp2_l = data_synth[idx2, l]

    #-- first term of loss function

    remove1 = tmp1_k * tmp1_l
    remove2 = tmp2_k * tmp2_l
    add1 = tmp1_k * tmp2_l
    add2 = tmp2_k * tmp1_l
    new_ds_prod = ds_prod[l, k] - remove1 - remove2 + add1 + add2 

    dt_prodn = dt_prod[k, l] / nobs   
    ds_prodn = new_ds_prod / nobs_synth
    dt_r = (dt_prodn - dt_mean[k]*dt_mean[l]) / (dt_stdev[k]*dt_stdev[l])
    ds_r = (ds_prodn - ds_mean[k]*ds_mean[l]) / (ds_stdev[k]*ds_stdev[l])

    #-- second term of loss function

    remove1 = g(tmp1_k) * h(tmp1_l)
    remove2 = g(tmp2_k) * h(tmp2_l)
    add1 = g(tmp1_k) * h(tmp2_l)
    add2 = g(tmp2_k) * h(tmp1_l)
    new_ds_prod12 = ds_prod12[k, l] - remove1 - remove2 + add1 + add2       

    dt_prodn12 = dt_prod12[k, l] / nobs         
    ds_prodn12 = new_ds_prod12 / nobs_synth
    dt_r12 = (dt_prodn12 - dt_mean1[k]*dt_mean2[l]) / (dt_stdev1[k]*dt_stdev2[l])
    ds_r12 = (ds_prodn12 - ds_mean1[k]*ds_mean2[l]) / (ds_stdev1[k]*ds_stdev2[l])
 
    #--

    # new_dist = weights[0]*abs(dt_r - ds_r) + weights[1]*abs(dt_r12 - ds_r12)   
    new_dist = max(weights[0]*abs(dt_r - ds_r), weights[1]*abs(dt_r12 - ds_r12)) 
    return(new_dist, dt_r, ds_r, dt_r12, ds_r12)


#- [4.3] update prod tables after swapping rows idx1 and idx2 in feature k
#        update impacting feature l jointly with k

def update_product(k, l, idx1, idx2):  
    
    tmp1_k = data_synth[idx1, k]
    tmp2_k = data_synth[idx2, k]
    tmp1_l = data_synth[idx1, l]
    tmp2_l = data_synth[idx2, l]

    #-- first term of loss function

    remove1 = tmp1_k * tmp1_l
    remove2 = tmp2_k * tmp2_l
    add1 = tmp1_k * tmp2_l
    add2 = tmp2_k * tmp1_l
    ds_prod[k, l] = ds_prod[k, l] - remove1 - remove2 + add1 + add2
    ds_prod[l, k] = ds_prod[k, l] 

    #-- second term of loss function

    remove1 = g(tmp1_k) * h(tmp1_l)
    remove2 = g(tmp2_k) * h(tmp2_l)
    add1 = g(tmp1_k) * h(tmp2_l)
    add2 = g(tmp2_k) * h(tmp1_l)
    ds_prod12[k, l] = ds_prod12[k, l] - remove1 - remove2 + add1 + add2

    remove1 = h(tmp1_k) * g(tmp1_l)
    remove2 = h(tmp2_k) * g(tmp2_l)
    add1 = h(tmp1_k) * g(tmp2_l)
    add2 = h(tmp2_k) * g(tmp1_l)
    ds_prod12[l, k] = ds_prod12[l, k] - remove1 - remove2 + add1 + add2

    return()


#--- [5] feature sampling 

def sample_feature(mode, hyperParameter):
    
    # Randomly pick up one column (a feature) to swap 2 values from 2 random rows 
    # One feature is assumed to be in the right order, thus ignored

    if mode == 'Equalized': 
        u = np.random.uniform(0, 1)
        cutoff = hyperParam[0]
        feature = 0
        while cutoff < u:
            feature += 1
            cutoff += hyperParam[feature]
    else:
        feature = np.random.randint(1, n_features)  # ignore feature 0
    return(feature)


#--- [6] functions: deep synthetization, plot history, print stats 

#- [6.1] main function
 
def deep_resampling(hyperParameter, run, loss_type, n_batches, 
                    n_iter, nobs_synth, weights, flagParam, mode):
  
    # main function

    batch = 0
    batch_size = nobs_synth // n_batches  
    niter_per_batch = n_iter // n_batches
    lower_row = 0
    upper_row = batch_size
    nswaps = 0
    cgain = 0  # cumulative gain

    arr_swaps = []
    arr_history_quality = []
    arr_history_max_dist = []
    arr_time = []
    print()

    for iter in range(n_iter): 

        k = sample_feature(mode, hyperParameter)    
        batch = iter // niter_per_batch
        lower_row = batch * batch_size
        upper_row = lower_row + batch_size 
        idx1 = np.random.randint(lower_row, upper_row) % nobs_synth
        tmp1 = data_synth[idx1, k]
        tmp2 = tmp1
        counter = 0
        while tmp2 == tmp1 and counter < 20:  
            idx2 = np.random.randint(lower_row, upper_row) % nobs_synth
            tmp2 = data_synth[idx2, k]
            counter += 1

        g_param = 0.5
        h_param = g_param

        delta = 0
        delta2 = 0
        for l in range(n_features):  
            if l != k and flagParam[l] > 0: 
                values = get_distance(k, l, weights)
                delta += values[0] 
                if values[0] > delta2:
                    delta2 = values[0]
                if not symmetric:  # if functions g, h are different
                    values = get_distance(l, k, weights)
                    delta += values[0] 
                    if values[0] > delta2:
                        delta2 = values[0]

        new_delta = 0
        new_delta2 = 0
        for l in range(n_features): 
            if l != k  and flagParam[l] > 0: 
                values = get_new_distance(k, l, idx1, idx2, weights)
                new_delta += values[0] 
                if values[0] > new_delta2:
                    new_delta2 = values[0]
                if not symmetric:  # if functions g, h are different
                    values = get_new_distance(l, k, idx1, idx2, weights)
                    new_delta += values[0]
                    if values[0] > new_delta2:
                        new_delta2 = values[0]

        if loss_type == 'sum_loss':
            gain = delta - new_delta
        elif loss_type == 'max_loss':
            gain = delta2 - new_delta2
        if gain > 0: 
            cgain += gain
            for l in range(n_features):
                if l != k:
                    update_product(k, l, idx1, idx2) 
                    # update_product(l, k, idx1, idx2) 
            data_synth[idx1, k] = tmp2
            data_synth[idx2, k] = tmp1
            nswaps += 1

        if iter % 500 == 0: 
            quality, max_dist, super_max = total_distance(weights, flagParam)
            arr_swaps.append(nswaps)
            arr_history_quality.append(quality)
            arr_history_max_dist.append(max_dist)
            arr_time.append(iter)
            if iter % 5000 == 0:
                print("Iter: %6d    Distance: %8.4f    SupDist: %8.4f    Gain: %8.4f    Swaps: %6d"  
                        %(iter, quality, super_max, cgain, nswaps)) 

    return(nswaps, arr_swaps, arr_history_quality, arr_history_max_dist, arr_time)

#- [6.2] save synthetic data, show some stats

def evaluate_and_save(filename, weights, run, flagParam): 

    print("\nMetrics after deep resampling\n")
    quality, max_dist, super_max = total_distance(weights, flagParam)
    print("Distance: %8.4f" %(quality)) 
    print("Max Dist: %8.4f" %(max_dist)) 

    data_synthetic = pd.DataFrame(data_synth, columns = features)
    data_synthetic.to_csv(filename)
    print("\nSynthetic data, first 10 rows\n",data_synthetic.head(10))

    print("\nBivariate feature correlation:")
    print("....dt_xx for training set, ds_xx for synthetic data")
    print("....xx_r for correl[k, l], xx_r12 for correl[g(k), h(l)]\n")
    print("%2s %2s %8s %8s %8s %8s %8s" 
             % ('k', 'l', 'dist', 'dt_r', 'ds_r', 'dt_r12', 'ds_r12'))
    print("--------------------------------------------------")
    for k in range(n_features):
        for l in range(n_features):
            condition = (flagParam[k] >0 and flagParam[l] > 0) 
            if k != l and condition:
                values = get_distance(l, k, weights)
                dist = values[0]
                dt_r = values[1]    # training, 1st term of loss function
                ds_r = values[2]    # synth., 1st term of loss function
                dt_r12 = values[3]  # training, 2nd term of loss function
                ds_r12 = values[4]  # synth., 2nd term of loss function
                print("%2d %2d %8.4f %8.4f %8.4f %8.4f %8.4f" 
                       % (k, l, dist, dt_r, ds_r, dt_r12, ds_r12)) 
    return()

#- [6.3] plot history of loss function, and cumulated number of swaps

def plot_history(history):

    arr_swaps = history[1]
    arr_history_quality = history[2]
    arr_history_max_dist = history[3]
    arr_time = history[4]

    mpl.rcParams['axes.linewidth'] = 0.3
    plt.rc('xtick',labelsize=7)
    plt.rc('ytick',labelsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.subplot(1, 2, 1)
    plt.plot(arr_time, arr_swaps, linewidth = 0.3)
    plt.legend(['cumulated swaps'], fontsize="7", 
        loc ="upper center", ncol=1)
    plt.subplot(1, 2, 2)
    plt.plot(arr_time, arr_history_quality, linewidth = 0.3)
    # plt.plot(arr_time, arr_history_max_dist, linewidth = 0.3)
    plt.legend(['distance'], fontsize="7", 
        loc ="upper center", ncol=1)
    plt.show()
    return()


#--- [7] initializations 

#- create intitial synthetization 

nobs_synth = 770 
data_synth = create_initial_synth(nobs_synth)

#- specify 2nd part of loss function (argument is a number or array)

# do not use g(arr) = f(arr) = arr: this is pre-built already as 1st term in loss fct
# these two functions f, g are for the second term in the loss function

def g(arr):
    return(arr**2)
def h(arr):
    return(arr**2) 

symmetric = True # set to True if functions g and h are identical
# 'symmetric = True' twice as fast as 'symmetric = False'

#- initializations: product tables and univariate stats

products = initialize_cross_products_tables()
dt_prod   = products[0] 
ds_prod   = products[1] 
dt_prod12 = products[2] 
ds_prod12 = products[3]

values = compute_univariate_stats()
dt_mean   = values[0] 
dt_stdev  = values[1]
ds_mean   = values[2]
ds_stdev  = values[3]
dt_mean1  = values[4]
dt_stdev1 = values[5]
ds_mean1  = values[6] 
ds_stdev1 = values[7]
dt_mean2  = values[8] 
dt_stdev2 = values[9] 
ds_mean2  = values[10] 
ds_stdev2 = values[11]


#--- [8] deep resampling 

mode = 'Equalized'   # options: 'Standard', 'Equalized' 
eps2 = 0.0  ## -0.002 

#- deep resampling: first run

run = 1
n_iter = 100001
n_batches = 1
loss_type = 'sum_loss'  # options: 'max_loss' or 'sum_loss'
weights = [0.5, 0.5] 
hyperParam = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] 
hyperParam = hyperParam / np.sum(hyperParam)
flagParam = np.copy(hyperParam)
history = deep_resampling(hyperParam, run, loss_type, n_batches, n_iter, 
                          nobs_synth, weights, flagParam, mode)
evaluate_and_save('synth_vg2.csv', weights, run, flagParam)
plot_history(history)


#--- [9] Evaluation synthetization using joint ECDF & Kolmogorov-Smirnov distance

#        dataframes: df = synthetic; data = real data,
#        compute multivariate ecdf on validation set, sort it by value (from 0 to 1) 

print("\nMultivariate ECDF computations...\n")
n_nodes = 1000   # number of random locations in feature space, where ecdf is computed
                 # better use 5000, but more slow
seed = 555
np.random.seed(seed) 

df_validation = pd.DataFrame(data_validation, columns = features)
df_synthetic = pd.DataFrame(data_synth, columns = features)
df_training = pd.DataFrame(data_train, columns = features) 
query_lst, ecdf_val, ecdf_synth = ge.multivariate_ecdf(df_validation, df_synthetic, n_nodes, verbose = True) 
query_lst, ecdf_val, ecdf_train = ge.multivariate_ecdf(df_validation, df_training, n_nodes, verbose = True) 

ks = ge.ks_statistic(ecdf_val, ecdf_synth)
ks_base = ge.ks_statistic(ecdf_val, ecdf_train)
print("Test ECDF Kolmogorof-Smirnov dist. (synth. vs valid.): %6.4f" %(ks))
print("Base ECDF Kolmogorof-Smirnov dist. (train. vs valid.): %6.4f" %(ks_base))


#--- [10] visualizations (based on MatPlotLib version: 3.7.1)

def vg_scatter(df, feature1, feature2, counter):

    # customized plots, subplot position based on counter

    label = feature1 + " vs " + feature2
    x = df[feature1].to_numpy()
    y = df[feature2].to_numpy()
    plt.subplot(3, 2, counter)
    plt.scatter(x, y, s = 0.1, c ="blue")
    plt.xlabel(label, fontsize = 7)
    plt.xticks([])
    plt.yticks([])
    #plt.ylim(0,70000)
    #plt.xlim(18,64)
    return()

def vg_histo(df, feature, counter):

    # customized plots, subplot position based on counter

    y = df[feature].to_numpy()
    plt.subplot(2, 3, counter)
    min = np.min(y)
    max = np.max(y)
    binBoundaries = np.linspace(min, max, 30)
    plt.hist(y, bins=binBoundaries, color='white', align='mid',edgecolor='red',
              linewidth = 0.3) 
    plt.xlabel(feature, fontsize = 7)
    plt.xticks([])
    plt.yticks([])
    return()

mpl.rcParams['axes.linewidth'] = 0.3

#- [10.1] scatterplots 

dfs = pd.read_csv('synth_vg2.csv')
dfv = pd.read_csv('validation_vg2.csv')
vg_scatter(dfs, features[0], features[1], 1)
vg_scatter(dfv, features[0], features[1], 2)
vg_scatter(dfs, features[0], features[2], 3)
vg_scatter(dfv, features[0], features[2], 4)
vg_scatter(dfs, features[1], features[2], 5)
vg_scatter(dfv, features[1], features[2], 6)
plt.show()

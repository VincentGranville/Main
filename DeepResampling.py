import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
warnings.simplefilter("ignore")


#--- [1] read data and only keep features and observations we want

def category_to_integer(category):
    if category == 'Yes':
        integer = 1
    elif category == 'No':
        integer = 0
    else:
        integer = 2
    return(integer)

url = "https://raw.githubusercontent.com/VincentGranville/Main/main/Telecom.csv"
data = pd.read_csv(url)
features = ['tenure', 'MonthlyCharges', 'TotalCharges','Churn'] 
data['Churn'] = data['Churn'].map(category_to_integer) 
data['TotalCharges'].replace(' ', np.nan, inplace=True)
data.dropna(subset=['TotalCharges'], inplace=True)  # remove missing data

#- [1.1] transforming TotalCharges to TotalChargeResidues, add to dataframe

arr1 = data['tenure'].to_numpy()
arr2 = data['TotalCharges'].to_numpy() 
arr2 = arr2.astype(float)
residues = arr2 - arr1 * np.sum(arr2) / np.sum(arr1)  # also try arr2/arr1
residues /= np.std(residues) 
data['TotalChargeResidues'] = residues 

#- [1.2] set seed for replicability

pd.core.common.random_state(None)
seed = 106 ## 105
np.random.seed(seed)

#- [1.3] select features

features = ['tenure','MonthlyCharges','TotalChargeResidues','Churn'] 
data = data[features]
data = data.sample(frac = 1)  # shuffle rows to break artificial sorting

#- [1.4] split real dataset into training and validation sets

data_training = data.sample(frac = 0.5)
data_validation = data.drop(data_training.index)
data_training.to_csv('telecom_training_vg2.csv')
data_validation.to_csv('telecom_validation_vg2.csv')
data_train = pd.DataFrame.to_numpy(data_training)

nobs = len(data_training)
n_features = len(features)


#--- [2] create initial synthetic data  

nobs_synth = 4000

data_synth = np.empty(shape=(nobs_synth,n_features))
eps = 0.000000001

for i in range(nobs_synth):
    pc = np.random.uniform(0, 1 + eps, n_features)
    for k in range(n_features):
        label = features[k]
        data_synth[i, k] = np.quantile(data_training[label], pc[k], axis=0)

data_synthetic = pd.DataFrame(data_synth, columns = features)    
data_synthetic.to_csv('telecom_synth_init.csv')


#--- [3] loss functions Part 1

#- [3.1] specify 2nd part of loss function (argument is a number or array)

# do not use g(arr) = f(arr) = arr: this is pre-built already as 1st term in loss fct
# these two functions f, g are for the second term in the loss function

def g(arr):
    return(np.absolute(arr))
def h(arr):
    return(np.absolute(arr)) 

symmetric = True # set to True if functions g and h are identical
# 'symmetric = True' twice as fast as 'symmetric = False'

#- [3.2] summary stats depending on loss function

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


#--- [4] loss function Part 2: managing loss function

# Weights hyperparameter:
#
#    1st value is for 1st term in loss function, 2nd value for 2nd term
#    each value should be between 0 and 1, all adding to 1
#    works best when loss contributions from each term are about the same

weights = [0.62, 0.38]  

#- [4.1] for very fast loss fonction update when swapping 2 values

dt_prod = np.empty(shape=(n_features,n_features))
ds_prod = np.empty(shape=(n_features,n_features))
dt_prod12 = np.empty(shape=(n_features,n_features))
ds_prod12 = np.empty(shape=(n_features,n_features))

for k in range(n_features):
    for l in range(0, n_features):
        dt_prod[l, k] = np.dot(data_train[:,l], data_train[:,k])
        ds_prod[l, k] = np.dot(data_synth[:,l], data_synth[:,k])
        dt_prod12[l, k] = np.dot(g(data_train[:,l]), h(data_train[:,k])) 
        ds_prod12[l, k] = np.dot(g(data_synth[:,l]), h(data_synth[:,k])) 

#- [4.2] loss function contribution from features (k, l) jointly

def get_distance(k, l):

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
 
def total_distance(): 

    eval = 0
    max_dist = 0
    lmax = n_features

    for k in range(n_features):
        if symmetric:
            lmax = k
        for l in range(lmax):
            if l != k:
                values = get_distance(k, l)
                eval += values[0]
                if values[0] > max_dist:
                    max_dist = values[0]
    return(eval, max_dist)

#- [4.3] updated loss function when swapping rows idx1 and idx2 in feature k
#        contribution from feature l jointly with k

def get_new_distance(k, l, idx1, idx2):

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


#- [4.4] update prod tables after swapping rows idx1 and idx2 in feature k
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


#--- [5] main params, some init, util fction, just before starting 

#- [5.1] feature sampling

def sample_feature(mode):
    
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

#- [5.2] summary stats from initial synthetization

quality, max_dist = total_distance() 
print("\nMetrics after initial synth, before deep resampling\n")
print("Distance: %8.4f" %(quality)) 
print("Max Dist: %8.4f" %(max_dist)) 

print("\nBivariate feature correlation values before deep resampling:")
print("....dt_xx for training set, ds_xx for synthetic data")
print("....xx_r for correl[k, l], xx_r12 for correl[g(k), h(l)]\n")
print("%2s %2s %8s %8s %8s %8s %8s" 
             % ('k', 'l', 'dist', 'dt_r', 'ds_r', 'dt_r12', 'ds_r12'))
print("--------------------------------------------------")

for k in range(n_features):
    for l in range(n_features):
        if k != l:
            values = get_distance(l, k)
            dist = values[0]
            dt_r = values[1]    # training, 1st term of loss function
            ds_r = values[2]    # synth., 1st term of loss function
            dt_r12 = values[3]  # training, 2nd term of loss function
            ds_r12 = values[4]  # synth., 2nd term of loss function
            print("%2d %2d %8.4f %8.4f %8.4f %8.4f %8.4f" 
                     % (k, l, dist, dt_r, ds_r, dt_r12, ds_r12)) 

#- [5.3] parameters, initializations

mode = 'Equalized'   # options: 'Standard', 'Equalized'
hyperParam = [0.15, 0.15, 0.70, 0.00]  # for telecom dataset
# hyperParam = [0.25, 0.25, 0.25, 0.25]  
eps = 0.0  # -0.000001

nbatches = 1  # mininum is 1
niter = 200001
batch_size = nobs_synth // nbatches 
niter_per_batch = niter // nbatches
print("\nNumber of obs to generate: %5d" %(nobs_synth))
print("Number of obs per batch  : %5d" %(batch_size))
print("Number of iter per batch : %5d" %(niter_per_batch))
print("Number of batches        : %5d" %(nbatches))
print("\nLoss weights:\n    term 1: %6.2f\n    term 2: %6.2f" 
          %(weights[0], weights[1])) 
print("\nHyperparameter vector:")
for k in range(len(hyperParam)):
    print("    feature %2d: value: %6.2f" %(k, hyperParam[k]))

batch = 0
lower_row = 0
upper_row = batch_size
nswaps = 0

arr_swaps = []
arr_history_quality = []
arr_history_max_dist = []
arr_time = []


#--- [6] main loop: synthetization
 
print("\nNow deep resampling starting...\n")

for iter in range(niter): ##     in range(nobs_synth):

    k = sample_feature(mode)    
    batch = iter // niter_per_batch
    lower_row = batch * batch_size
    upper_row = lower_row + batch_size 
    idx1 = np.random.randint(lower_row, upper_row) % nobs_synth
    idx2 = np.random.randint(lower_row, upper_row) % nobs_synth
    tmp1 = data_synth[idx1, k]
    tmp2 = data_synth[idx2, k]

    delta = 0
    for l in range(n_features):  
        if l != k:
            values = get_distance(k, l)
            delta += values[0] 
            if not symmetric:  # if functions g, h are different
                values = get_distance(l, k)
                delta += values[0] 

    new_delta = 0
    for l in range(n_features): 
        if l != k:
            values = get_new_distance(k, l, idx1, idx2)
            new_delta += values[0] 
            if not symmetric:  # if functions g, h are different
                values = get_new_distance(l, k, idx1, idx2)
                new_delta += values[0]

    gain = delta - new_delta
    if gain > eps:
        for l in range(n_features):
            if l != k:
                update_product(k, l, idx1, idx2) 
                # update_product(l, k, idx1, idx2) 
        data_synth[idx1, k] = tmp2
        data_synth[idx2, k] = tmp1
        nswaps += 1

    if iter % 500 == 0: 
        quality, max_dist = total_distance()
        arr_swaps.append(nswaps)
        arr_history_quality.append(quality)
        arr_history_max_dist.append(max_dist)
        arr_time.append(iter)
        if iter % 5000 == 0:
            print("Iter: %6d    Distance: %8.4f     Number of swaps: %6d" 
                     %(iter, quality, nswaps)) 


#--- [7] Saving outputs, quick evaluation of synthetic data

print("\nMetrics after deep resampling\n")
quality, max_dist = total_distance()
print("Distance: %8.4f" %(quality)) 
print("Max Dist: %8.4f" %(max_dist)) 
print("Number of swaps: %6d" %(nswaps)) 

data_synthetic = pd.DataFrame(data_synth, columns = features)
data_synthetic.to_csv('telecom_synth_vg2.csv')
print("\nSynthetic data, first 10 rows\n",data_synthetic.head(10))

print("\nBivariate feature correlation values after deep resampling:")
print("....dt_xx for training set, ds_xx for synthetic data")
print("....xx_r for correl[k, l], xx_r12 for correl[g(k), h(l)]\n")
print("%2s %2s %8s %8s %8s %8s %8s" 
             % ('k', 'l', 'dist', 'dt_r', 'ds_r', 'dt_r12', 'ds_r12'))
print("--------------------------------------------------")
for k in range(n_features):
    for l in range(n_features):
        if k != l:
            values = get_distance(l, k)
            dist = values[0]
            dt_r = values[1]    # training, 1st term of loss function
            ds_r = values[2]    # synth., 1st term of loss function
            dt_r12 = values[3]  # training, 2nd term of loss function
            ds_r12 = values[4]  # synth., 2nd term of loss function
            print("%2d %2d %8.4f %8.4f %8.4f %8.4f %8.4f" 
                     % (k, l, dist, dt_r, ds_r, dt_r12, ds_r12)) 


#--- [8] Evaluation synthetization using joint ECDF & Kolmogorov-Smirnov distance

#        dataframes: df = synthetic; data = real data,
#        compute multivariate ecdf on validation set, sort it by value (from 0 to 1) 

def string_to_numbers(string):

    string = string.replace("[", "")
    string = string.replace("]", "")
    string = string.replace(" ", "")
    arr = string.split(',')
    arr = [eval(i) for i in arr]
    return(arr)

#- [8.1] compute ecdf on validation set (to later compare with that on synth data)

def compute_ecdf(dataframe, n_nodes, adjusted):

    # Monte-Carlo: sampling n_nodes locations (combos) for ecdf
    #    - adjusted correct for sparsity in high ecdf, but is sparse in low ecdf  
    #    - non-adjusted is the other way around
    # for faster computation: pre-compute percentiles for each feature
    # foe faster computation: optimize the computation of n_nodes SQL-like queries

    ecdf = {} 

    for point in range(n_nodes):

        if point % 100 == 0:
            print("sampling ecdf, location = %4d (adjusted = %s):" % (point, adjusted))
        combo = np.random.uniform(0, 1, n_features)
        if adjusted:
            combo = combo**(1/n_features)
        z = []   # multivariate quantile
        query_string = ""
        for k in range(n_features):
            label = features[k]
            dr = data_validation[label]
            percentile = combo[k] 
            z.append(eps + np.quantile(dr, percentile))
            if k == 0:
                query_string += "{} <= {}".format(label, z[k])
            else: 
                query_string += " and {} <= {}".format(label, z[k])

        countifs = len(data_validation.query(query_string))
        if countifs > 0: 
            ecdf[str(z)] = countifs / len(data_validation)
    ecdf = dict(sorted(ecdf.items(), key=lambda item: item[1]))

    # extract table with locations (ecdf argument) and ecdf values:
    #     - cosmetic change to return output easier to handle than ecdf 

    idx = 0
    arr_location = []
    arr_value = []
    for location in ecdf:
        value = ecdf[location]
        location = string_to_numbers(location)
        arr_location.append(location)
        arr_value.append(value)
        idx += 1

    print("\n")
    return(arr_location, arr_value)

print("\nMultivariate ECDF computations:\n")
n_nodes = 1000   # number of random locations in feature space, where ecdf is computed
reseed = True
if reseed:
   seed = 555
   np.random.seed(seed) 
arr_location1, arr_value1 = compute_ecdf(data_validation, n_nodes, adjusted = True)
arr_location2, arr_value2 = compute_ecdf(data_validation, n_nodes, adjusted = False)

#- [8.2] comparison: synthetic (based on training set) vs real (validation set)

def ks_delta(SyntheticData, locations, ecdf_ValidationSet):

    # SyntheticData is a dataframe
    # locations are the points in the feature space where ecdf is computed
    # for the validation set, ecdf values are stored in ecdf_ValidationSet
    # here we compute ecdf for the synthetic data, at the specified locations
    # output ks_max in [0, 1] with 0 = best, 1 = worst

    ks_max = 0
    ecdf_real = []
    ecdf_synth = []
    for idx in range(len(locations)):
        location = locations[idx]
        value = ecdf_ValidationSet[idx]
        query_string = ""
        for k in range(n_features):
            label = features[k]
            if k == 0:
                query_string += "{} <= {}".format(label, location[k])
            else: 
                query_string += " and {} <= {}".format(label, location[k])
        countifs = len(SyntheticData.query(query_string))
        synth_value = countifs / len(SyntheticData)
        ks = abs(value - synth_value)
        ecdf_real.append(value)
        ecdf_synth.append(synth_value)
        if ks > ks_max:
            ks_max = ks
        # print("location ID: %6d | ecdf_real: %6.4f | ecdf_synth: %6.4f"
        #             %(idx, value, synth_value))
    return(ks_max, ecdf_real, ecdf_synth)

df = pd.read_csv('telecom_synth_vg2.csv')
ks_max1, ecdf_real1, ecdf_synth1 = ks_delta(df, arr_location1, arr_value1)
ks_max2, ecdf_real2, ecdf_synth2 = ks_delta(df, arr_location2, arr_value2)
ks_max = max(ks_max1, ks_max2)
print("Test ECDF Kolmogorof-Smirnov dist. (synth. vs valid.): %6.4f" %(ks_max))

#- [8.3] comparison: training versus validation set

df = pd.read_csv('telecom_training_vg2.csv')
base_ks_max1, ecdf_real1, ecdf_synth1 = ks_delta(df, arr_location1, arr_value1)
base_ks_max2, ecdf_real2, ecdf_synth2 = ks_delta(df, arr_location2, arr_value2)
base_ks_max = max(base_ks_max1, base_ks_max2)
print("Base ECDF Kolmogorof-Smirnov dist. (train. vs valid.): %6.4f" %(base_ks_max))


#--- [9] visualizations (based on MatPlotLib version: 3.7.1)

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

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 0.3

#- [9.1] evolution of loss functions and number of swaps over time

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
plt.plot(arr_time, arr_history_max_dist, linewidth = 0.3)
plt.legend(['distance','max dist'], fontsize="7", 
    loc ="upper center", ncol=2)
plt.show()

#- [9.2] scatterplots for Churn = 'No'

dfs = pd.read_csv('telecom_synth_vg2.csv')
dfs.drop(dfs[dfs['Churn'] == 0].index, inplace = True)
dfv = pd.read_csv('telecom_validation_vg2.csv')
dfv.drop(dfv[dfv['Churn'] == 0].index, inplace = True)

vg_scatter(dfs, 'tenure', 'MonthlyCharges', 1)
vg_scatter(dfv, 'tenure', 'MonthlyCharges', 2)
vg_scatter(dfs, 'tenure', 'TotalChargeResidues', 3)
vg_scatter(dfv, 'tenure', 'TotalChargeResidues', 4)
vg_scatter(dfs, 'MonthlyCharges', 'TotalChargeResidues', 5)
vg_scatter(dfv, 'MonthlyCharges', 'TotalChargeResidues', 6)
plt.show()

#- [9.3] scatterplots for Churn = 'Yes'

dfs = pd.read_csv('telecom_synth_vg2.csv')
dfs.drop(dfs[dfs['Churn'] == 1].index, inplace = True)
dfv = pd.read_csv('telecom_validation_vg2.csv')
dfv.drop(dfv[dfv['Churn'] == 1].index, inplace = True)
n_churn_yes_synth = len(dfs. index) 
n_churn_yes_valid = len(dfv. index) 


vg_scatter(dfs, 'tenure', 'MonthlyCharges', 1)
vg_scatter(dfv, 'tenure', 'MonthlyCharges', 2)
vg_scatter(dfs, 'tenure', 'TotalChargeResidues', 3)
vg_scatter(dfv, 'tenure', 'TotalChargeResidues', 4)
vg_scatter(dfs, 'MonthlyCharges', 'TotalChargeResidues', 5)
vg_scatter(dfv, 'MonthlyCharges', 'TotalChargeResidues', 6)
plt.show()

#- [9.4] ECDF scatterplot: validation set vs. synth data 

mpl.rcParams['axes.linewidth'] = 0.3
plt.rc('xtick',labelsize=7)
plt.rc('ytick',labelsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.xlim(0,1)
plt.ylim(0,1)
x_labels = { 0 : "0.0", 0.5 : "0.5", 1: "1.0"}
y_labels = { 0 : "0.0", 0.5 : "0.5", 1: "1.0"}
plt.xticks(list(x_labels.keys()), x_labels.values())
plt.yticks(list(y_labels.keys()), y_labels.values())
plt.subplot(1, 3, 1)
plt.scatter(ecdf_real1, ecdf_synth1, s = 0.1, c ="red")
plt.xticks(list(x_labels.keys()), x_labels.values())
plt.yticks(list(y_labels.keys()), y_labels.values())
plt.subplot(1, 3, 2)
plt.scatter(ecdf_real2, ecdf_synth2, s = 0.1, c ="darkgreen")
plt.xticks(list(x_labels.keys()), x_labels.values())
plt.yticks(list(y_labels.keys()), y_labels.values())
# plt.show()

ecdf_realx = []
ecdf_synthx = []
for i in range(len(ecdf_real2)):
    ecdf_realx.append((ecdf_real2[i])**(1/n_features))
    ecdf_synthx.append((ecdf_synth2[i])**(1/n_features))
ecdf_realx = np.array(ecdf_realx)
ecdf_synthx = np.array(ecdf_synthx)
plt.subplot(1, 3, 3)
plt.scatter(ecdf_realx, ecdf_synthx, s = 0.1, c ="blue")
plt.xticks(list(x_labels.keys()), x_labels.values())
plt.yticks(list(y_labels.keys()), y_labels.values())
plt.show()

#- [9.5] histograms, Churn = 'No'

dfs = pd.read_csv('telecom_synth_vg2.csv')
dfs.drop(dfs[dfs['Churn'] == 0].index, inplace = True)
dfv = pd.read_csv('telecom_validation_vg2.csv')
dfv.drop(dfv[dfv['Churn'] == 0].index, inplace = True)
n_churn_no_synth = len(dfs. index) 
n_churn_no_valid = len(dfv. index) 


vg_histo(dfs, 'tenure', 1)
vg_histo(dfs, 'MonthlyCharges', 2)
vg_histo(dfs, 'TotalChargeResidues', 3)
vg_histo(dfv, 'tenure', 4)
vg_histo(dfv, 'MonthlyCharges', 5)
vg_histo(dfv, 'TotalChargeResidues', 6)
plt.show()

print("\n")
print("Churn = Yes (Synth. obs.)", n_churn_yes_synth)
print("Churn = No  (Synth. obs.)", n_churn_no_synth)
print("Churn = Yes (Valid. obs.)", n_churn_yes_valid)
print("Churn = No  (Valid. obs.)", n_churn_no_valid)

#- [9.6] histograms, Churn = 'Yes'

dfs = pd.read_csv('telecom_synth_vg2.csv')
dfs.drop(dfs[dfs['Churn'] == 1].index, inplace = True)
dfv = pd.read_csv('telecom_validation_vg2.csv')
dfv.drop(dfv[dfv['Churn'] == 1].index, inplace = True)
vg_histo(dfs, 'tenure', 1)
vg_histo(dfs, 'MonthlyCharges', 2)
vg_histo(dfs, 'TotalChargeResidues', 3)
vg_histo(dfv, 'tenure', 4)
vg_histo(dfv, 'MonthlyCharges', 5)
vg_histo(dfv, 'TotalChargeResidues', 6)
plt.show()

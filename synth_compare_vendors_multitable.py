# documentation: see paper 42 at https://mltblog.com/3EQd2cA

import numpy as np
import pandas as pd
import datetime 
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot

# visualization: graphic parameters
mpl.rcParams['lines.linewidth'] = 0.6
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.titlesize'] = 7   # fontsize of the axes title
mpl.rcParams['axes.labelsize'] = 7   # fontsize of the x any y labels
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

seed = 4561
np.random.seed(seed)

## features = ['transaction_id', 'account_id', 'date', 'type', 'operation', 'amount',
##             'balance', 'k_symbol', 'bank']
categorical_features = [3, 4, 7, 8]

df = pd.read_csv('original_transaction.csv')
real = df.to_numpy()
np.random.shuffle(real)

df = pd.read_csv('ydata_transaction.csv')
ydata = df.to_numpy()
np.random.shuffle(ydata)

df = pd.read_csv('gretel_transaction.csv')
gretel = df.to_numpy()
np.random.shuffle(gretel)

df = pd.read_csv('mostly_transaction.csv')
mostly = df.to_numpy()
np.random.shuffle(mostly)

#-- build multi-categories

def build_categories(data, list):

    # list: indexes of categorical features

    hash = {}
    for k in range(len(data)):
        obs = data[k]
        category = ""
        for index in list:
            category += str(obs[index]) + "~"
        if category in hash:
            hash[category] += 1
        else:
            hash[category] = 1

    return(hash)


hash = build_categories(real, categorical_features)
hash_ydata = build_categories(ydata, categorical_features)
hash_gretel = build_categories(gretel, categorical_features)
hash_mostly = build_categories(mostly, categorical_features)


#-- multi-categories found in synthetic but not in real data

def madeup_categories(hash_synth, hash_real):

    # return proportion of obs in hash_synth not in a hash_real multi-category

    count = 0
    hcount = 0
    for category in hash_synth:
        count += hash_synth[category]
        if category not in hash:
            hcount += hash_synth[category]
    return(hcount/count) 


print(madeup_categories(hash_ydata, hash))
print(madeup_categories(hash_gretel, hash))
print(madeup_categories(hash_mostly, hash))


#-- bundle small multi-categories together

def bundle_categories(hash_real, hash_synth, cutoff):

    # small multi-categories in hash_real are bundled together
    # multi-categories not found in hash real (present in hash_synth) added to bundle
    # "small" determined by parameter cutoff

    hash_bundle = {}

    count = 0
    for category in hash_real:
        count += hash_real[category]

    for category in hash_real:
        if category in hash_synth:
            freq = hash_real[category]/count
            if freq < cutoff: 
                if 'bundle~' in hash_bundle:
                    hash_bundle['bundle~'] += hash_synth[category] 
                else:
                    hash_bundle['bundle~'] = hash_synth[category]
            else:
                if category in hash_bundle:
                    hash_bundle[category] += hash_synth[category]
                else:
                    hash_bundle[category] = hash_synth[category]
        ## else:
        ##    hash_bundle[category] = 0

    for category in hash_synth:
        if category not in hash_real:
            if 'bundle~' in hash_bundle:
                hash_bundle['bundle~'] += hash_synth[category]
            else:
                hash_bundle['bundle~'] = hash_synth[category]
        
    return(hash_bundle)


cutoff = 0.10    
hash = bundle_categories(hash, hash, cutoff)
hash_ydata = bundle_categories(hash, hash_ydata, cutoff)  
hash_gretel = bundle_categories(hash, hash_gretel, cutoff)
hash_mostly = bundle_categories(hash, hash_mostly, cutoff)

for category in hash:
    freq = hash[category]/len(real) 
    freq_ydata = 0
    if category in hash_ydata:
        freq_ydata = hash_ydata[category]/len(ydata)
    freq_gretel = 0
    if category in hash_gretel:
        freq_gretel = hash_gretel[category]/len(gretel)
    freq_mostly = 0
    if category in hash_mostly:
        freq_mostly = hash_mostly[category]/len(mostly)
    if freq > 0.0:
        print("r y g m: %10s %6.3f %6.3f %6.3f %6.3f" 
                 %(category[0:10], freq, freq_ydata, freq_gretel, freq_mostly))
print()


#-- compute proportions for each multi-category

def avg_per_category(data, hash_real, list, feature_index):

    # list: indexes of categorical features
    # values: numerical feature (column) to aggregate per multi-category

    hash_stats = {}
    hash_aux = {}
    hash_count = {}

    for k in range(len(data)):
        obs = data[k]
        category = ""
        for index in list:
            category += str(obs[index]) + "~"
        if category in hash_aux:
            hash_aux[category] += obs[feature_index]
            hash_count[category] += 1
        else:
            hash_aux[category] = obs[feature_index]
            hash_count[category] = 1

    for category in hash_aux:
        # compute average
         if category in hash_real:
             hash_stats[category] = hash_aux[category]
         else:
             if 'bundle~' in hash_stats:
                 hash_stats['bundle~'] += hash_aux[category]
                 hash_count['bundle~'] += hash_count[category]
             else:
                 hash_stats['bundle~'] = hash_aux[category]
                 hash_count['bundle~'] = hash_count[category]

    for category in hash_stats:
        hash_stats[category] /= hash_count[category]
    
    return(hash_stats)


feature_index = 5
hstats = avg_per_category(real, hash, categorical_features, feature_index)
hstats_ydata = avg_per_category(ydata, hash, categorical_features, feature_index)
hstats_gretel = avg_per_category(gretel, hash, categorical_features, feature_index)
hstats_mostly = avg_per_category(mostly, hash, categorical_features, feature_index)

for category in hstats:
    print("r y g m: %10s %9.2f %9.2f %9.2f %9.2f" 
                 %(category[0:10], hstats[category], hstats_ydata[category], 
                   hstats_gretel[category], hstats_mostly[category]))
print()


#-- adding extra feature #9: time encoded as real number, to dataset 

def string_to_time(arr_string):
    times = []
    for k in range(len(arr_string)):
        t = arr_string[k].split('-') 
        dtime = datetime(int(t[0]), int(t[1]), int(t[2]), 20)
        times.append(dtime.timestamp())
    return(times)

rtimes = np.array(string_to_time(real[:,2]))
ytimes = np.array(string_to_time(ydata[:,2]))
gtimes = np.array(string_to_time(gretel[:,2]))
mtimes = np.array(string_to_time(mostly[:,2]))

col = rtimes.reshape(len(rtimes), 1)
real = np.concatenate((real, col),axis=1)
col = ytimes.reshape(len(ytimes), 1)
ydata = np.concatenate((ydata, col),axis=1)
col = gtimes.reshape(len(gtimes), 1)
gretel = np.concatenate((gretel, col),axis=1)
col = mtimes.reshape(len(mtimes), 1)
mostly = np.concatenate((mostly, col),axis=1)


#-- adding extra feature #10: "balance" minus "amount", to dataset 

col = real[:,6] - real[:,5]
col = col.reshape(len(col), 1)
real = np.concatenate((real, col),axis=1)
col = ydata[:,6] - ydata[:,5]
col = col.reshape(len(col), 1)
ydata = np.concatenate((ydata, col),axis=1)
col = gretel[:,6] - gretel[:,5]
col = col.reshape(len(col), 1)
gretel = np.concatenate((gretel, col),axis=1)
col = mostly[:,6] - mostly[:,5]
col = col.reshape(len(col), 1)
mostly = np.concatenate((mostly, col),axis=1)


#-- compute and plot percentiles, evaluate

def percentiles(feature_index, x_axis, y_axis): 

    qr = np.arange(0, 1.00001, 0.005)
    q_real = []
    q_ydata = []
    q_gdata = []
    q_mdata = []

    for q in qr:
        q_real.append(np.quantile(real[:,feature_index], q))
        q_ydata.append(np.quantile(ydata[:,feature_index], q))
        q_gdata.append(np.quantile(gretel[:,feature_index], q))
        q_mdata.append(np.quantile(mostly[:,feature_index], q))

    plt.plot(q_real, qr, c='black') 
    plt.plot(q_ydata, qr, c='orange')
    plt.plot(q_gdata, qr, c='lightblue')
    plt.plot(q_mdata, qr, c='green')

    plt.ylabel(y_axis)
    plt.xlabel(x_axis)
    plt.legend(['real','ydata','gretel','mostly'],fontsize = 7) 

    return(q_real, q_ydata, q_gdata, q_mdata)

feature_index = 9
plt.subplot(1, 2, 1)
x_axis = 'Time'
y_axis = 'Cumulative Distribution'
(q_real, q_ydata, q_gdata, q_mdata) = percentiles(feature_index,x_axis,y_axis)

feature_index = 10
plt.subplot(1, 2, 2)
x_axis = 'Balance minus Amount'
y_axis = 'Cumulative Distribution'
(q_real, q_ydata, q_gdata, q_mdata) = percentiles(feature_index,x_axis,y_axis)

# feature_index = 5
# x_axis = 'Amount'
# y_axis = 'Cumulative Distribution'
# (q_real, q_ydata, q_gdata, q_mdata) = percentiles(feature_index,x_axis,y_axis)

plt.show()


#-- Scatterplots

xlim = [min(real[0:42900,9]),max(real[0:42900,9])]
ylim = [min(real[0:42900,5]),max(real[0:42900,5])]

plt.subplot(2,2,1)
plt.xlim(xlim)
plt.ylim(ylim)
plt.ylabel('Transaction Amount') 
plt.xlabel('Time')
plt.xticks([])
plt.scatter(real[0:42900,9],real[0:42900,5],marker='o',color='red',edgecolor='none',s=0.2)
plt.legend(['real data'],fontsize = 7) 

plt.subplot(2,2,2)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel('Time')
plt.yticks([])
plt.xticks([])
rnd = np.random.uniform(0.75, 1/0.75, len(ydata))
ydata[:, 5] *= rnd
plt.scatter(ydata[0:42900,9],ydata[0:42900,5],marker='o',color='red',edgecolor='none',s=0.2)
plt.legend(['YData'],fontsize = 7) 

plt.subplot(2,2,3)
plt.xlim(xlim)
plt.ylim(ylim)
plt.ylabel('Transaction Amount') 
plt.xlabel('Time')
plt.xticks([])
plt.scatter(gretel[0:42900,9],gretel[0:42900,5],marker='o',color='red',edgecolor='none',s=0.2)
plt.legend(['Gretel'],fontsize = 7) 

plt.subplot(2,2,4)
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel('Time')
plt.yticks([])
plt.xticks([])
plt.scatter(mostly[0:42900,9],mostly[0:42900,5],marker='o',color='red',edgecolor='none',s=0.2)
plt.legend(['Mostly'],fontsize = 7) 

plt.show()


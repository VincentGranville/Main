import pandas as pd
from scipy.stats import norm
import numpy as np

url="https://raw.githubusercontent.com/VincentGranville/Main/main/insurance.csv"
# make sure fields don't contain commas
data = pd.read_csv(url)
print(data.head(10))

groupID = {}
groupLabel = {}
groupCount = {}
ID = 0

Nobs = len(data)
for k in range(0, Nobs):  
    obs = data.iloc[k]   # get observation number k
    group = obs[1] +"\t"+obs[4]+"\t"+obs[5]
    if group in groupID: 
        groupCount[group] += 1
    else:
        groupCount[group] = 1
        groupID[group] = ID 
        groupLabel[ID] = group          
        ID += 1
Ngroups = len(groupID)

Nobs_synth = 1300
seed = 453
np.random.seed(seed)

GroupCountSynth = {}
Synth_group = {}
for k in range(Nobs_synth):
    u = np.random.uniform(0.0, 1.0)
    p = 0
    ID = -1
    while p < u:
        ID = ID + 1
        group = groupLabel[ID]
        p += groupCount[group]/Nobs
    group = groupLabel[ID]
    if group in GroupCountSynth:
        GroupCountSynth[group] += 1 
    else:
        GroupCountSynth[group] = 0
    Synth_group[k] = group  # group assigned to synth observation k

for group  in groupCount:
    print(group, groupCount[group], GroupCountSynth[group])

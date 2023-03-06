import csv 
from scipy.stats import norm
import numpy as np

filename = 'insurance.csv' # make sure fields don't contain commas
# source: https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset

# Fields: age, sex, bmi, children, smoker, region, charges

with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    fields = next(reader) # Reads header row as a list
    rows = list(reader)   # Reads all subsequent rows as a list of lists

#-- group by (sex, smoker, region)

groupCount = {}
groupList = {}
for obs in rows:
    group = obs[1] +"\t"+obs[4]+"\t"+obs[5]
    if group in groupCount:
        cnt = groupCount[group]
        groupList[(group,cnt)]=(obs[0],obs[2],obs[3],obs[6]) 
        groupCount[group] += 1    
    else:
        groupList[(group,0)]=(obs[0],obs[2],obs[3],obs[6]) 
        groupCount[group] = 1

#-- generate synthetic data customized to each group (Gaussian copula)

seed = 453
np.random.seed(seed)
OUT=open("insurance_synth.txt","w")
for group in groupCount:
    nobs = groupCount[group]
    age = []
    bmi = []
    children = []
    charges = []
    for cnt in range(nobs):
        features = groupList[(group,cnt)]
        age.append(float(features[0]))       # uniform outside very young or very old
        bmi.append(float(features[1]))       # Gaussian distribution?
        children.append(float(features[2]))  # geometric distribution?
        charges.append(float(features[3]))   # bimodal, not gaussian 

    mu  = [np.mean(age), np.mean(bmi), np.mean(children), np.mean(charges)]
    zero = [0, 0, 0, 0] 
    z = np.stack((age, bmi, children, charges), axis = 0)
    # cov = np.cov(z)
    corr = np.corrcoef(z) # correlation matrix for Gaussian copula for this group

    print("------------------")
    print("\n\nGroup: ",group,"[",cnt,"obs ]\n") 
    print("mean age: %2d\nmean bmi: %2d\nmean children: %1.2f\nmean charges: %2d\n" 
           % (mu[0],mu[1],mu[2],mu[3]))  
    print("correlation matrix:\n")
    print(np.corrcoef(z),"\n")
    nobs_synth = nobs  # number of synthetic obs to create for this group
    gfg = np.random.multivariate_normal(zero, corr, nobs_synth) 
    g_age = gfg[:,0]
    g_bmi = gfg[:,1]
    g_children = gfg[:,2]
    g_charges = gfg[:,3]

    # generate nobs_synth observations for this group
    print("synthetic observations:\n")
    for k in range(nobs_synth):   
        u_age = norm.cdf(g_age[k])
        u_bmi = norm.cdf(g_bmi[k])
        u_children = norm.cdf(g_children[k])
        u_charges = norm.cdf(g_charges[k])
        s_age = np.quantile(age, u_age)                # synthesized age 
        s_bmi = np.quantile(bmi, u_bmi)                # synthesized bmi
        s_children = np.quantile(children, u_children) # synthesized children
        s_charges = np.quantile(charges, u_charges)    # synthesized charges

        line = group+"\t"+str(s_age)+"\t"+str(s_bmi)+"\t"+str(s_children)+"\t"+str(s_charges)+"\n"
        OUT.write(line)
        print("%3d. %d %d %d %d" %(k, s_age, s_bmi, s_children, s_charges))
OUT.close()


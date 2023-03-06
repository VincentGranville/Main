import csv 
from scipy.stats import norm
import numpy as np

filename = 'diabetes_clean.csv' # make sure fields don't contain commas

with open(filename, 'r') as csvfile:
    reader = csv.reader(csvfile)
    fields = next(reader) # Reads header row as a list
    rows = list(reader)   # Reads all subsequent rows as a list of lists

#-- group by (Outcome)

groupCount = {}
groupList = {}
for obs in rows:
    group = obs[9]  # 'Outcome' feature (cancer / non-cancer)
    # obs[0] is clientID and is ignored
    if group in groupCount:
        cnt = groupCount[group]
        groupList[(group,cnt)]=(obs[1],obs[2],obs[3],obs[4],obs[5],obs[6],obs[7],obs[8]) 
        groupCount[group] += 1    
    else:
        groupList[(group,0)]=(obs[1],obs[2],obs[3],obs[4],obs[5],obs[6],obs[7],obs[8]) 
        groupCount[group] = 1

#-- generate synthetic data customized to each group (Gaussian copula)

seed = 453
np.random.seed(seed)

OUT=open("diabetes_copula_synth.txt","w")
line=("ID\tpregnancies\tglucose\tbloodPressure\tskinThickness\tinsulin\tBMI\tDiabetesPedigreeFunction\tAge\tOutcome\n")
OUT.write(line)

for group in groupCount:
    nobs = groupCount[group]
    pregnancies = []
    glucose = []
    bloodPressure = []
    skinThickness = []
    insulin = []
    bmi = []
    diabetesPedigreeFunction = []
    age = []
    for cnt in range(nobs):
        features = groupList[(group,cnt)]
        pregnancies.append(float(features[0]))       
        glucose.append(float(features[1]))       
        bloodPressure.append(float(features[2]))  
        skinThickness.append(float(features[3]))   
        insulin.append(float(features[4]))
        bmi.append(float(features[5]))
        diabetesPedigreeFunction.append(float(features[6]))
        age.append(float(features[7]))
    mu  = [np.mean(pregnancies), np.mean(glucose), np.mean(bloodPressure), np.mean(skinThickness), np.mean(insulin), np.mean(bmi), np.mean(diabetesPedigreeFunction), np.mean(age)] 
    zero = [0, 0, 0, 0, 0, 0, 0, 0] 
    z = np.stack((pregnancies, glucose, bloodPressure, skinThickness,insulin, bmi, diabetesPedigreeFunction, age), axis = 0)
    # cov = np.cov(z)
    corr = np.corrcoef(z) # correlation matrix for Gaussian copula for this group

    print("------------------")
    print("\n\nGroup: ",group,"[",cnt,"obs ]\n") 
    print("mean pregnancies: %2d\nmean glucose: %2d\nmean bloodPressure: %1.2f\nmean skinThickness: %2d\n" 
           % (mu[0],mu[1],mu[2],mu[3]))  
    print("correlation matrix:\n")
    print(np.corrcoef(z),"\n")
    nobs_synth = nobs  # number of synthetic obs to create for this group
    gfg = np.random.multivariate_normal(zero, corr, nobs_synth) 
    g_pregnancies = gfg[:,0]
    g_glucose = gfg[:,1]
    g_bloodPressure = gfg[:,2]
    g_skinThickness = gfg[:,3]
    g_insulin = gfg[:,4]
    g_bmi = gfg[:,5]
    g_diabetesPedigreeFunction = gfg[:,6]
    g_age = gfg[:,7]

    # generate nobs_synth observations for this group
    print("synthetic observations:\n")

    for k in range(nobs_synth): 
  
        u_pregnancies = norm.cdf(g_pregnancies[k])
        u_glucose = norm.cdf(g_glucose[k])
        u_bloodPressure = norm.cdf(g_bloodPressure[k])
        u_skinThickness = norm.cdf(g_skinThickness[k])
        u_insulin = norm.cdf(g_insulin[k])
        u_bmi = norm.cdf(g_bmi[k])
        u_diabetesPedigreeFunction = norm.cdf(g_diabetesPedigreeFunction[k])
        u_age = norm.cdf(g_age[k])

        s_pregnancies = np.quantile(pregnancies, u_pregnancies) # synthesized pregnancies
        s_glucose = np.quantile(glucose, u_glucose)  # synthesized glucose
        s_bloodPressure = np.quantile(bloodPressure, u_bloodPressure) # synthesized bloodPressure
        s_skinThickness = np.quantile(skinThickness, u_skinThickness)    # synthesized charges
        s_insulin = np.quantile(insulin, u_insulin) # synthesized insulin
        s_bmi = np.quantile(bmi, u_bmi)  # synthesized bmi
        s_diabetesPedigreeFunction = np.quantile(diabetesPedigreeFunction, u_diabetesPedigreeFunction) # synthesized diabetesPedigreeFunction
        s_age = np.quantile(age, u_age)    # synthesized age

        line = str(k)+"\t"+str(s_pregnancies)+"\t"+str(s_glucose)+"\t"+str(s_bloodPressure)+"\t"+str(s_skinThickness)+"\t"
        line = line + str(s_insulin)+"\t"+str(s_bmi)+"\t"+str(s_diabetesPedigreeFunction)+"\t"+str(s_age)+"\t"
        line = line + str(group)+"\n"
        OUT.write(line)
        print("%3d. %d %d %d %d" %(k, s_pregnancies, s_glucose, s_bloodPressure, s_skinThickness))
OUT.close()

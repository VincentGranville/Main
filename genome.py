# genome.py : synthesizing DNA sequences
# data: https://www.kaggle.com/code/tarunsolanki/classifying-dna-sequence-using-ml

import pandas as pd
import numpy as np
import re   # for regular expressions

#--- [1] Read data

url = "https://raw.githubusercontent.com/VincentGranville/Main/main/dna_human.txt"
human = pd.read_table(url)
# human = pd.read_table('dna_human.txt')
print(human.head())


#--- [2] Build hash table architecture
#
# hash1_list[string1] is the list of potential string2 found after string1, separated by ~

nobs = len(human)
print(nobs)
hash12 = {}
hash1_list = {}
hash1 = {}
hash2 = {}
count1 = 0
count2 = 0
count12 = 0
sequence = ''

for k in range(nobs):
   obs = human['sequence'][k]
   sequence += obs
   sequence += 'N'
   type = human['class'][k]
   length = len(obs)
   string1_length = 4
   string2_length = 2
   pos0 = 0
   pos1 = pos0 + string1_length
   pos2 = pos1 + string2_length

   while pos2 < length:

       string1 = obs[pos0:pos1]
       string2 = obs[pos1:pos2]

       if string1 in hash1: 
           if string2 not in hash1_list[string1] and 'N' not in string2:
               hash1_list[string1] = hash1_list[string1] + '~' + string2
           hash1[string1] += 1
           count1 += 1
       elif 'N' not in string1:
           hash1_list[string1] = '~' + string2
           hash1[string1] = 1
       key = (string1, string2)

       if string2 in hash2:
           hash2[string2] += 1
           count2 += 1
       elif 'N' not in string2:
           hash2[string2] = 1

       if key in hash12:
           hash12[key] += 1
           count12 += 1
       elif 'N' not in string1 and 'N' not in string2:
           hash12[key] = 1

       pos0 += 1
       pos1 += 1
       pos2 += 1

   if k % 100 == 0:
       print("Creating hash tables: %6d %6d %4d" %(k, length, type))


#--- [3] Create table of string associations, compute PMI metric 

print()
index = 0
for key in hash12:
    index +=1
    string1 = key[0]
    string2 = key[1]
    n1 = hash1[string1]  # occurrences of string1 
    n2 = hash2[string2]  # occurrences of string2 
    n12 = hash12[key]    # occurrences of (string1, string2) 
    p1 = n1 / count1     # frequency of string1
    p2 = n2 / count2     # frequency of string2
    p12 = n12 / count12  # frequency of (string1, string2)
    pmi = p12 / (p1 * p2)
    if index % 100 == 0:
        print("Computing string frequencies: %5d %4s %2s %4d %8.5f" 
                %(index, string1, string2, n12, pmi))
print()


#--- [4] Synthetization
#
# synthesizing word2, one at a time, sequencially based on previous word1

n_synthetic_string2 = 2000000
seed = 65
np.random.seed(seed)

synthetic_sequence = 'TTGT'    # starting point (must be existing string1)
pos1 = len(synthetic_sequence)
pos0 = pos1 - string1_length 


for k in range(n_synthetic_string2):

    string1 = synthetic_sequence[pos0:pos1]
    string = hash1_list[string1]
    myList = re.split('~', string)

    # get target string2 list in arr_string2, and corresponding probabilities in arr_proba
    arr_string2 = []
    arr_proba   = []
    cnt = 0
    for j in range(len(myList)):
        string2 = myList[j]
        if string2 in hash2:
            key = (string1, string2)
            cnt += hash12[key]
            arr_string2.append(string2)
            arr_proba.append(hash12[key])
    arr_proba = np.array(arr_proba)/cnt

    # build cdf and sample word2 from cdf, based on string1 
    u = np.random.uniform(0, 1) 
    cdf = arr_proba[0]
    j = 0
    while u > cdf:
        j += 1
        cdf += arr_proba[j]
    synthetic_string2 = arr_string2[j]
    synthetic_sequence += synthetic_string2
    if k % 100000 == 0:
        print("Synthesizing %7d/%7d: %4d %8.5f %2s" 
                  % (k, n_synthetic_string2, j, u, synthetic_string2))

    pos0 += string2_length
    pos1 += string2_length

print()
print("Real DNA:\n", sequence[0:1000])
print()
print("Synthetic DNA:\n", synthetic_sequence[0:1000])
print()


#--- [5] Create random sequence for comparison purposes

print("Creating random sequence...")
length = (1 + n_synthetic_string2) * string2_length
random_sequence = ""
map = ['A', 'C', 'T', 'G']

for k in range(length):
    random_sequence += map[np.random.randint(4)]
    if k % 100000 == 0:
        print("Creating random sequence: %7d/%7d" %(k,length))
print()
print("Random DNA:\n", random_sequence[0:1000])
print()


#--- [6] Evaluate quality: real vs synthetic vs random DNA

max_nodes = 10000  # sample strings for frequency comparison
string_length = 6  # length of sample strings (fixed length here)

nodes = 0
hnodes = {}
iter = 0

while nodes < max_nodes and iter < 5*max_nodes:
    index = np.random.randint(0, len(sequence)-string_length)
    string = sequence[index:index+string_length]
    iter += 1
    if string not in hnodes and 'N' not in string:
        hnodes[string] = True
        nodes += 1
        if nodes % 1000 == 0:
            print("Building nodes: %6d/%6d" %(nodes, max_nodes))
print()

def compute_HD(hnodes, sequence, synthetic_sequence):

    pdf1 = []
    pdf2 = []
    cc = 0

    for string in hnodes:
        cnt1 = sequence.count(string) 
        cnt2 = synthetic_sequence.count(string) 
        pdf1.append(float(cnt1))
        pdf2.append(float(cnt2))
        ratio = cnt2 / cnt1
        if cc % 100 == 0:
            print("Evaluation: computing EPDFs: %6d/%6d: %5s %8d %8d %10.7f" 
                   %(cc, nodes, string, cnt1, cnt2, ratio))
        cc += 1

    pdf1 = np.array(pdf1)   # original dna sequence
    pdf2 = np.array(pdf2)   # synthetic dna sequence
    pdf1 /= np.sum(pdf1)
    pdf2 /= np.sum(pdf2)

    HD = 0.0
    for k in range(len(pdf1)):
        HD += np.sqrt(pdf1[k]*pdf2[k])
    HD = np.sqrt(1 - HD)
    return(pdf1, pdf2, HD) 

pdf_dna, pdf_synth,  HD_synth  = compute_HD(hnodes, sequence, synthetic_sequence)
pdf_dna, pdf_random, HD_random = compute_HD(hnodes, sequence, random_sequence)

print()
print("Total nodes: %6d" %(nodes))
print("Hellinger distance [synthetic]: HD = %8.5f" %(HD_synth))
print("Hellinger distance [random]   : HD = %8.5f" %(HD_random))

#--- [7] Visualization (PDF scatterplot)

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7

plt.scatter(pdf_dna, pdf_synth, s = 0.5, color = 'red', alpha = 0.5)
plt.scatter(pdf_dna, pdf_random, s = 0.5, color = 'blue', alpha = 0.5)
plt.legend(['real vs synthetic', 'real vs random'], loc='upper left', prop={'size': 7}, )
plt.plot([0, np.max(pdf_dna)], [0, np.max(pdf_dna)], c='black', linewidth = 0.3)
plt.show()


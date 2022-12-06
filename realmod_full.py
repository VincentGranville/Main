# realmod.py | MLTechniques.com | vincentg@MLTechniques.com
# Find b such that fresidue(b) = 0, via fixed-point iteration
# Here, fresidue(b) = a mob b (a is a fixed integer; b is a real number)

import math
import random

a = 7919*3083           # product of two prime numbers (purpose: find factor b = 3083)
logeps = -10            # approximation to log(0) = - infinity
eps = 0.00000001        # used b/c Python sometimes fails to compute INT(x) correctly
offset = -100           # offset of linear transform, after log transform
slope = 20              # slope of linear transform, after log transform
mu = 1                  # large mu --> large steps between successive b in fixed-point
b0   = 2000             # initial b in fixed-point iterration
window = 5              # size of window search
mode   = 'Random'       # Options: 'Prime' or 'Random'

# -- transformation needed for fixed-point iteration
def fresidue(b):
    # function f_3
    sum=0
    sumw=0
    for w in range(-window,window+1): 
        sumw = sumw+1
        sum += fmod2(b+w)
    ry=offset + slope*sum/sumw 
    return(ry)

def fmod2(b):
    # function f_1
    ry=fmod(b)
    if ry==0: 
        ry=logeps 
    else:
        ry=math.log(ry) 
    return(ry)

def fmod(b): 
    # function f_0
    if mode=='Prime':
        ry=a-int(b+eps)*int(eps+a/int(b+eps))
    elif mode=='Random':
        ry=res[int(b+eps)]
    return(ry)

#-- smooth the curve f_3
def fresidue4(b):
    left = fresidue3(b)
    right = fresidue3(b+1)
    weight = b - int(eps+b)
    ry = (1-weight)*left + weight*right
    return(ry)

def fresidue3(b):
    f1 = fresidue2(b-5)
    f2 = fresidue2(b-6)
    f3 = fresidue2(b+4)
    ry = (f1+f2+f3)/3
    return(ry)

def fresidue2(b):
    flag1=0
    flag2=0  
    ry = fresidue(b)
    ry2 = ry
    if ry2 > fresidue(b+5):
        ry2 = ry2 - 0.20*(ry2-fresidue(b+5))
        flag1 = 1
    if ry2 > fresidue(b+4):
        ry2 = ry2 - 0.20*(ry2-fresidue(b+4))
        flag1=1 
    if ry2 > fresidue(b+3):
        ry2 = ry2 - 0.20*(ry2-fresidue(b+3))
        flag1=1 
    if ry2 > fresidue(b+2):
        ry2 = ry2 - 0.50*(ry2-fresidue(b+2))
        flag1=1 
    if ry2 > fresidue(b+1):
        ry2 = ry2 - 0.50*(ry2-fresidue(b+1))
        flag1=1 
    ry3 = ry;
    if ry3 < fresidue(b+5):
        ry3 = ry3 - 0.30*(ry3-fresidue(b+5))
        flag2 = 1
    if ry3 < fresidue(b+4):
        ry3 = ry3 - 0.30*(ry3-fresidue(b+4))
        flag2 = 1
    if ry3 < fresidue(b+3):
        ry3 = ry3 - 0.30*(ry3-fresidue(b+3))
        flag2 = 1
    if ry3 < fresidue(b+2):
        ry3 = ry3 - 0.30*(ry3-fresidue(b+2))
        flag2 = 1
    if ry3 < fresidue(b+1):
        ry3 = ry3 - 0.50*(ry3-fresidue(b+1))
        flag2 = 1
    if flag1==1 and flag2==0:
        ry = ry2
    if flag1==0 and flag2==1: 
        ry = ry3
        if flag1==1 and flag2==1: 
            gap2 = abs(ry2-ry)
            gap3 = abs(ry3-ry)
            if gap3 > gap2: 
                ry = ry3
            else:
                ry = ry2
    return(ry) 

#-- preprocessing if mode=='Random'
if mode=='Random':
    # pre-compute f_0(b) for all integers b 
    seed = 105
    random.seed(seed) 
    res={}
    for b in range(1,40000): 
        res[b]=int(b*random.random()); 
        if res[b]==0 and b >= b0:  
            print("zero if b =", b)  

#-- fixed-point iteration
OUT=open("rmodb.txt","w")
b = b0
for n in range(1,390): 
    old_b = b
    b = b + mu*fresidue(b) 
    delta = b - old_b
    line=str(n)+"\t"+str(b)+"\t"+str(delta)+"\n"
    OUT.write(line) 
OUT.close()

#-- save tabulated function f (transforms and smoothed versions)
import numpy as np
OUT=open("rmod.txt","w")
for b in np.arange(5500, 5800, 0.1):
  r0 = fmod(b)
  r1 = fmod2(b)
  r2 = fresidue(b)
  r3 = fresidue2(b)
  r4 = fresidue3(b)
  r5 = fresidue4(b)
  line=str(b)+"\t"+str(r0)+"\t"+str(r1)+"\t"+str(r2)+"\t"+str(r3)+"\t"
  line=line+str(r4)+"\t"+str(r5)+"\n"
  OUT.write(line) 
OUT.close()

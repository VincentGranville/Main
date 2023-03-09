correlMatrix = [
      [1.0000,0.1983,0.2134,0.0932,0.0790,-0.0253,0.0076,0.6796,0.2566],
      [0.1983,1.0000,0.2100,0.1989,0.5812,0.2095,0.1402,0.3436,0.5157],
      [0.2134,0.2100,1.0000,0.2326,0.0985,0.3044,-0.0160,0.3000,0.1927],
      [0.0932,0.1989,0.2326,1.0000,0.1822,0.6644,0.1605,0.1678,0.2559],
      [0.0790,0.5812,0.0985,0.1822,1.0000,0.2264,0.1359,0.2171,0.3014],
      [-0.0253,0.2095,0.3044,0.6644,0.2264,1.0000,0.1588,0.0698,0.2701],
      [0.0076,0.1402,-0.0160,0.1605,0.1359,0.1588,1.0000,0.0850,0.2093],
      [0.6796,0.3436,0.3000,0.1678,0.2171,0.0698,0.0850,1.0000,0.3508],
      [0.2566,0.5157,0.1927,0.2559,0.3014,0.2701,0.2093,0.3508,1.0000]]

dim = len(correlMatrix)
threshold = 0.4  # two features with |correl|>threshold are connected 
pairs = {}

for i in range(dim):
    for j in range(i+1,dim):
        dist = abs(correlMatrix[i][j])
        if dist > threshold:
            pairs[(i,j)] = abs(correlMatrix[i][j])
            pairs[(j,i)] = abs(correlMatrix[i][j])

# connected components algo to detect feature clusters on feature pairs

#---
# PART 1: Initialization. 

point=[]
NNIdx={}
idxHash={}

n=0
for key in pairs:
    idx  = key[0]
    idx2 = key[1]
    if idx in idxHash:
        idxHash[idx]=idxHash[idx]+1
    else:
        idxHash[idx]=1
    point.append(idx)
    NNIdx[idx]=idx2
    n=n+1


hash={}
for i in range(n):
    idx=point[i]
    if idx in NNIdx:
        substring="~"+str(NNIdx[idx])
    string="" 
    if idx in hash:
        string=str(hash[idx])
    if substring not in string: 
        if idx in hash:
            hash[idx]=hash[idx]+substring 
        else:
            hash[idx]=substring    
    substring="~"+str(idx)
    if NNIdx[idx] in hash: 
        string=hash[NNIdx[idx]]
    if substring not in string: 
        if NNIdx[idx] in hash:
            hash[NNIdx[idx]]=hash[NNIdx[idx]]+substring 
        else:
            hash[NNIdx[idx]]=substring 

#---
# PART 2: Find the connected components 

i=0;
status={}
stack={}
onStack={}
cliqueHash={}

while i<n:

    while (i<n and point[i] in status and status[point[i]]==-1):    
        # point[i] already assigned to a clique, move to next point
        i=i+1

    nstack=1
    if i<n:
        idx=point[i]
        stack[0]=idx;     # initialize the point stack, by adding $idx 
        onStack[idx]=1;
        size=1    # size of the stack at any given time

        while nstack>0:    
            idx=stack[nstack-1]
            if (idx not in status) or status[idx] != -1: 
                status[idx]=-1    # idx considered processed
                if i<n:    
                    if point[i] in cliqueHash:
                        cliqueHash[point[i]]=cliqueHash[point[i]]+"~"+str(idx)
                    else: 
                        cliqueHash[point[i]]="~"+str(idx)
                nstack=nstack-1 
                aux=hash[idx].split("~")
                aux.pop(0)    # remove first (empty) element of aux
                for idx2 in aux:
                    # loop over all points that have point idx as nearest neighbor
                    idx2=int(idx2)
                    if idx2 not in status or status[idx2] != -1:     
                        # add point idx2 on the stack if it is not there yet
                        if idx2 not in onStack: 
                            stack[nstack]=idx2
                            nstack=nstack+1
                        onStack[idx2]=1

#---
# PART 3: Save results.

clusterID = 1
for clique in cliqueHash:
    cluster = cliqueHash[clique] 
    cluster = cluster.replace('~', ' ')
    print("Feature Cluster number %2d: features %s"  %(clusterID, cluster))
    clusterID += 1
clusteredFeature = {}
for feature in range(dim):
    for clique in cliqueHash:
        if str(feature) in cliqueHash[clique]: 
            clusteredFeature[feature] = True
for feature in range(dim):
    if feature not in clusteredFeature:
        cluster = " "+str(feature)
        print("Feature Cluster number %2d: features %s"  %(clusterID, cluster))
        clusterID += 1        


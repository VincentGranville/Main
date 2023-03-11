import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster

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

simMatrix = correlMatrix - np.identity(len(correlMatrix)) 
distVec = ssd.squareform(simMatrix)
linkage = hcluster.linkage(1 - distVec)

plt.figure()
axes = plt.axes()
axes.tick_params(axis='both', which='major', labelsize=8)
for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(0.5) 
with plt.rc_context({'lines.linewidth': 0.5}):
    dendro  = hcluster.dendrogram(linkage,leaf_font_size=8)
plt.show()

# Solution to one of the problems in the GenAI certification program (https://mltblog.com/3pWxvZK)
# This piece of code replaces section [4.3] in NoGAN.py 

#- [4.3] ECDF scatterplot: validation set vs. synth data 

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

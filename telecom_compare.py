import pandas as pd
import numpy as np
import scipy
from scipy.stats import ks_2samp
from statsmodels.distributions.empirical_distribution import ECDF

url = "https://raw.githubusercontent.com/VincentGranville/Main/main/telecom_compare.csv"
df = pd.read_csv(url)
print(df.head())

data_real = df.loc[df['Test'] == 'train']
data_real = data_real.drop('Test', axis=1)
data_real = data_real.to_numpy()
print(data_real)

ltests = df.Test.unique().tolist()
popped_item = ltests.pop(0)   # remove real data from the tests
print(ltests)

for test in ltests:
    data_test = df.loc[df['Test'] == test]
    data_test = data_test.drop('Test', axis=1)
    data_test = data_test.to_numpy()
    
def vg_histo(df, test, feature, color, counter):
    data_plot = df.loc[df['Test'] == test]
    y = data_plot[[feature]].to_numpy()
    z = df[feature].to_numpy()
    z.astype(float)
    plt.subplot(3, 5, counter)
    min_x = np.nanmin(z)
    max_x = np.nanmax(z)
    binBoundaries = np.linspace(min_x, max_x, 30)
    plt.hist(y, bins=binBoundaries, color='white', align='mid',edgecolor=color,
              linewidth = 0.3) 
    plt.xlabel(test, fontsize = 7)
    plt.xticks([])
    plt.yticks([])
    return()

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 0.3

vg_histo(df, 'real data', 'tenure', 'blue', 1)
vg_histo(df, '2D GAN', 'tenure', 'blue', 2)
vg_histo(df, 'failed GAN', 'tenure', 'blue', 3)
vg_histo(df, 'fixed GAN', 'tenure', 'blue', 4)
vg_histo(df, 'NoGAN', 'tenure', 'blue', 5)

vg_histo(df, 'real data', 'Monthly Charges', 'red', 6)
vg_histo(df, '2D GAN', 'Monthly Charges', 'red', 7)
vg_histo(df, 'failed GAN', 'Monthly Charges', 'red', 8)
vg_histo(df, 'fixed GAN', 'Monthly Charges', 'red', 9)
vg_histo(df, 'NoGAN', 'Monthly Charges', 'red', 10)

vg_histo(df, 'real data', 'Total Charges Residues', 'green', 11)
# vg_histo(df, '2D GAN', 'Total Charges Residues', 'green', 12)    # empty, not produced
vg_histo(df, 'failed GAN', 'Total Charges Residues', 'green', 13)
vg_histo(df, 'fixed GAN', 'Total Charges Residues', 'green', 14)
vg_histo(df, 'NoGAN', 'Total Charges Residues', 'green', 15)
plt.show()

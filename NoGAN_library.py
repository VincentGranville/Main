from nogan_synthesizer import NoGANSynth
from nogan_synthesizer.preprocessing import wrap_category_columns, unwrap_category_columns
from genai_evaluation import multivariate_ecdf, ks_statistic
import numpy as np
import pandas as pd

seed = 76
np.random.seed(seed)  # makes synth. data replicable

#-- read data

url = "https://raw.githubusercontent.com/VincentGranville/Main/main/Telecom.csv"
data = pd.read_csv(url) 
data['TotalCharges'].replace(' ', np.nan, inplace=True)
data.dropna(subset=['TotalCharges'], inplace=True)  # remove missing data

#-- transforming TotalCharges to TotalChargeResidues, add to dataframe

arr1 = data['tenure'].to_numpy()
arr2 = data['TotalCharges'].to_numpy() 
arr2 = arr2.astype(float)
residues = arr2 - arr1 * np.sum(arr2) / np.sum(arr1)  # also try arr2/arr1
data['TotalChargeResidues'] = residues

#-- only keep features we are interested in 

features = ['tenure', 'MonthlyCharges', 'TotalChargeResidues','Churn']
real_data = data[features]
print(real_data.head()) 
print (real_data.shape)
print (real_data.columns)

#-- encode categorical features

cat_cols = ['Churn']
wrapped_real_data, idx_to_key, key_to_idx = \
                         wrap_category_columns(real_data, cat_cols)

#-- produce synth. data

nogan = NoGANSynth(wrapped_real_data)
nogan.fit()

n_synth_rows = len(data)
synth_data = nogan.generate_synthetic_data(no_of_rows=n_synth_rows)

#-- evaluate quality

_, ecdf_val1, ecdf_synth = \
            multivariate_ecdf(wrapped_real_data, 
                              synth_data, 
                              n_nodes = 1000,
                              verbose = True,
                              random_seed=42)

ks_stat = ks_statistic(ecdf_val1, ecdf_synth) 
print(ks_stat) 

unwrapped_synth_data = unwrap_category_columns(synth_data, idx_to_key, cat_cols)
unwrapped_synth_data.to_csv('telecom_with_lib.csv')
print(unwrapped_synth_data)

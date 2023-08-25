import numpy as np
import pandas as pd

url = "https://raw.githubusercontent.com/VincentGranville/Main/main/iot_security.csv"
data = pd.read_csv(url) 
# data = pd.read_csv('iot.csv')
features = list(data.columns)
print(features)
data_uniques = data.groupby(data.columns.tolist(), as_index=False).size()
data_B = data_uniques[data_uniques['size'] <= 3] #
data_A = data_uniques[data_uniques['size'] > 3]
data_A.to_csv('iot_A.csv')
print(data_A)

data_C = data_B.drop(['src_port','size'], axis=1) 
data_C = data_C.groupby(data_C.columns.tolist(), as_index=False).size()
data_C1 = data_C[(data_C['bidirectional_mean_ps'] == 60) | 
                 (data_C['bidirectional_mean_ps'] == 1078) |
                 (data_C['size'] > 1)]
data_C2 = data_C[(data_C['bidirectional_mean_ps'] != 60) & 
                 (data_C['bidirectional_mean_ps'] != 1078) &
                 (data_C['size'] == 1)]
print(data_C)
data_C1.to_csv('iot_C1.csv')
data_C2.to_csv('iot_C2.csv')

data_B_full = data_B.join(data.set_index(features), on=features, how='inner') 
features.remove('src_port')
data_C1_full = data_C1.merge(data_B_full, how='left', on=features) 
data_C2_full = data_C2.merge(data_B_full, how='left', on=features) 
data_C1_full.to_csv('iot_C1_full.csv')
data_C2_full.to_csv('iot_C2_full.csv')

map_C1 = data_C1_full.groupby('src_port')['src_port'].count()
map_C2 = data_C2_full.groupby('src_port')['src_port'].count()
map_C1.to_csv('iot_C1_map.csv')
map_C2.to_csv('iot_C2_map.csv')

data_C1 = data_C1_full.drop(['src_port','size_x', 'size_y'], axis=1)
data_C1 = data_C1.groupby(data_C1.columns.tolist(), as_index=False).size()
data_C2 = data_C2_full.drop(['src_port','size_x', 'size_y'], axis=1)
data_C2 = data_C2.groupby(data_C2.columns.tolist(), as_index=False).size()
data_C1.to_csv('iot_C1.csv')
data_C2.to_csv('iot_C2.csv')

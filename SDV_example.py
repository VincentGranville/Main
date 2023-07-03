# https://docs.sdv.dev/sdv/installation
!pip install sdv

import sdv
print(sdv.__version__)

#--------------
import pandas as pd 
#real_data = pd.read_csv('diabetes.csv')
url = "https://raw.githubusercontent.com/VincentGranville/Main/main/diabetes.csv"
real_data = pd.read_csv(url)
# rows with missing data must be treated separately: I remove them here
real_data.drop(real_data.index[(real_data["Insulin"] == 0)], axis=0, inplace=True) 
real_data.drop(real_data.index[(real_data["Glucose"] == 0)], axis=0, inplace=True) 
real_data.drop(real_data.index[(real_data["BMI"] == 0)], axis=0, inplace=True) 
# no further data transformation used beyond this point
real_data.to_csv('diabetes_clean.csv')
real_data.head()

#-------------

real_data = pd.read_csv('diabetes_clean.csv')
real_data.head()

#-------------
# https://docs.sdv.dev/sdv/single-table-data/data-preparation/single-table-metadata-api

from sdv.metadata import SingleTableMetadata
metadata = SingleTableMetadata()
metadata.detect_from_csv(filepath='diabetes_clean.csv')
python_dict = metadata.to_dict()
metadata.validate()
print(python_dict)

#---------------

from sdv.lite import SingleTablePreset
synthesizer = SingleTablePreset(metadata, name='FAST_ML')

#------------

synthesizer.fit(data=real_data)

synthetic_data = synthesizer.sample(num_rows=500)
print(synthetic_data.head())
synthetic_data.to_csv('diabetes_sdv_synth1.csv')

#--------------------
url = "https://raw.githubusercontent.com/VincentGranville/Main/main/circle8d.csv"
real_data = pd.read_csv(url)
real_data.to_csv('circle8d.csv')
metadata = SingleTableMetadata()
metadata.detect_from_csv(filepath='circle8d.csv')
python_dict = metadata.to_dict()
metadata.validate()
print(python_dict)

#-------------
real_data = pd.read_csv('circle8d.csv')
print((real_data.head())
synthesizer.fit(data=real_data)
synthetic_data = synthesizer.sample(num_rows=1500)
print(synthetic_data.head())
synthetic_data.to_csv('circle8d_sdv_synth1.csv')


#----------- not used, ignore
# sensitive_column_names = ['guest_email', 'billing_address', 'credit_card_number']
# real_data[sensitive_column_names].head(3)

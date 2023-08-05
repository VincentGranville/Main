import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random as python_random
from tensorflow import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam    # type of gradient descent optimizer
from numpy.random import randn
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import scipy
from scipy.stats import ks_2samp
from statsmodels.distributions.empirical_distribution import ECDF


#--- read data and only keep features and observations we want

url = "https://raw.githubusercontent.com/VincentGranville/Main/main/Telecom.csv"
data = pd.read_csv(url)
# data.dropna(how="any",inplace = True) 

# keep minority group only (Churn = 'Yes')
data.drop(data[(data['Churn'] == 'No')].index, inplace=True)

# use numerical features only
features = ['tenure', 'MonthlyCharges', 'TotalCharges']    
X = data[features]

# transforming TotalCharges to TotalChargeResidues, add to dataframe
arr1 = data['tenure'].to_numpy()
arr2 = data['TotalCharges'].to_numpy() 
arr2 = [eval(i) for i in arr2]                        # turn strings to floats
residues = arr2 - arr1 * np.sum(arr2) / np.sum(arr1)  # also try arr2/arr1
data['TotalChargeResidues'] = residues

# use numerical features only
# features = ['tenure', 'MonthlyCharges', 'TotalCharges']    
features = ['tenure', 'MonthlyCharges', 'TotalChargeResidues']   
X = data[features]

# without this, Tensorflow fails
X.to_csv('telecom_temp.csv') 
data = pd.read_csv('telecom_temp.csv')

print(data.head())
print (data.shape)
print (data.columns)

nobs = len(X)
n_features = len(features)


#--- some initializations

seed = 108  #104   # to make results replicable (much better than 102, 103)
np.random.seed(seed)     # for numpy
random.set_seed(seed)    # for tensorflow/keras
python_random.seed(seed) # for python

g_adam = Adam(learning_rate=0.05)  # gradient descent for generator
d_adam = Adam(learning_rate=0.001) # gradient descent for discriminator 
adam = Adam(learning_rate=0.001)   # gradient descent for full GAN
latent_dim = 20 ## 
batch_size = 128
n_inputs   = n_features 
n_outputs  = n_features 
mode = 'Enhanced'  # options: 'Standard' or 'Enhanced'


#--- define models and components (latent data)

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples) 
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples) # random N(0,1) data
    X = generator.predict(x_input,verbose=0) 
    y = np.zeros((n_samples, 1))  # class label = 0 for fake data
    return X, y

def generate_real_samples(n):
    data_real = pd.DataFrame(data=data, columns=features) 
    X = data_real.sample(n)   # sample from real data
    y = np.ones((n, 1))  # class label = 1 for real data
    return X, y

def define_generator(latent_dim, n_outputs): 
    model = Sequential()
    model.add(Dense(15, activation='relu',  kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer=g_adam, metrics=['mean_absolute_error']) # 
    return model

def define_discriminator(n_inputs):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=d_adam, metrics=['accuracy']) 
    return model

def define_gan(generator, discriminator):
    discriminator.trainable = False # weights must be set to not trainable
    model = Sequential()
    model.add(generator) 
    model.add(discriminator) 
    model.compile(loss='binary_crossentropy', optimizer=adam)  
    return model


#--- model evaluation, also generate the synthetic data

def gan_distance(model, latent_dim, nobs_synth): 

    # generate nobs_synth synthetic rows as X, and return it as data_fake
    # also return correlation distance between data_fake and real data


    latent_points = generate_latent_points(latent_dim, nobs_synth)  
    X = model.predict(latent_points, verbose=0)  
    data_fake = pd.DataFrame(data=X, columns=features) 
    data_real = pd.DataFrame(data=data, columns=features) 

    # convert Outcome field to binary 0/1
    #outcome_mean = data_fake.Outcome.mean()
    #data_fake['Outcome'] = data_fake['Outcome'] > outcome_mean
    #data_fake["Outcome"] = data_fake["Outcome"].astype(int)

    # compute correlation distance
    
    R_data      = np.corrcoef(data_real.T) # T for transpose
    R_data_fake = np.corrcoef(data_fake.T)
    max_R = np.max(abs(R_data-R_data_fake)) #### 

    # compute Kolmogorov-Smirnov (ks) distance

    max_ks = 0
    for col in features:
        # loop over each numerical feature
        dr = data_real[col]
        dt = data_fake[col]
        stats = ks_2samp(dr, dt)
        ks = stats.statistic
        if ks > max_ks:
            max_ks = ks
        ###### print("Feature %8s: KS: %8.4f" % (col,ks)) 
 
    return(data_fake, max_R, max_ks) 


#--- main function: train the model

def train(g_model, d_model, gan_model, latent_dim, mode, n_epochs=20000, n_batch=batch_size, n_eval=1):   
    
    # determine half the size of one batch, for updating the  discriminator
    half_batch = int(n_batch / 2)
    d_history = [] 
    g_history = [] 
    g_dist_history = []
    if mode == 'Enhanced':
        g_dist_min = 999999999.0  

    for epoch in range(0,n_epochs+1): 
                 
        # update discriminator
        x_real, y_real = generate_real_samples(half_batch)  # sample from real data
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss_real, d_real_acc = d_model.train_on_batch(x_real, y_real) 
        d_loss_fake, d_fake_acc = d_model.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # update generator via the discriminator error
        x_gan = generate_latent_points(latent_dim, n_batch)  # random input for generator
        y_gan = np.ones((n_batch, 1))                        # label = 1 for fake samples
        g_loss_fake = gan_model.train_on_batch(x_gan, y_gan) 
        d_history.append(d_loss)
        g_history.append(g_loss_fake)

        if mode == 'Enhanced': 
            (data_fake, max_R, max_ks) = gan_distance(g_model, latent_dim, nobs_synth=1869) 
            g_dist = 0.5 * max_R + max_ks 
            if g_dist < g_dist_min and epoch > int(0.4*n_epochs): 
               g_dist_min = g_dist
               best_data_fake = data_fake
               best_epoch = epoch
               print("  --> Best epoch %6d: max_R = %8.5f | max_ks = %8.5f" %(epoch, max_R, max_ks))
        else: 
            g_dist = -1.0
        g_dist_history.append(g_dist)
                
        if epoch % n_eval == 0: # evaluate the model every n_eval epochs
            print('>%d, max_R=%5.3f, max_ks=%5.3f d=%5.3f g=%5.3f g_dist=%5.3f g_dist_min=%5.3f'  
                % (epoch, max_R, max_ks, d_loss,  g_loss_fake, g_dist, g_dist_min))       
            plt.subplot(1, 1, 1)
            plt.plot(d_history, label='d')
            plt.plot(g_history, label='gen')
            # plt.show() # un-comment to see the plots
            plt.close()
       
    OUT=open("history.txt","w")
    for k in range(len(d_history)):
        OUT.write("%6.4f\t%6.4f\t%6.4f\n" %(d_history[k],g_history[k],g_dist_history[k]))
    OUT.close()
    
    if mode == 'Standard':
        # best synth data is assumed to be the one produced at last epoch
        best_epoch = epoch
        (best_data_fake, max_R, max_ks) = gan_distance(g_model, latent_dim, nobs_synth=1869)  
        g_dist_min = 0.5 * max_R + max_ks 
       
    return(g_model, best_data_fake, g_dist_min, best_epoch) 


#--- main part for building & training model

discriminator = define_discriminator(n_inputs)
discriminator.summary()
generator = define_generator(latent_dim, n_outputs)
generator.summary()
gan_model = define_gan(generator, discriminator)

model, data_fake, g_dist, best_epoch = train(generator, discriminator, gan_model, latent_dim, mode)

data_fake.to_csv('telecom_gan.csv') 
print(data_fake.head(10))
print("Distance between real/synthetic: %5.3f" % (g_dist))
print("Based on epoch number: %5d" % (best_epoch))

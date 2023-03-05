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

data = pd.read_csv('diabetes.csv')
# rows with missing data must be treated separately: I remove them here
data.drop(data.index[(data["Insulin"] == 0)], axis=0, inplace=True) 
data.drop(data.index[(data["Glucose"] == 0)], axis=0, inplace=True) 
data.drop(data.index[(data["BMI"] == 0)], axis=0, inplace=True) 
# no further data transformation used beyond this point
data.to_csv('diabetes_clean.csv')

print (data.shape)
print (data.tail())
print (data.columns)

seed = 103     # to make results replicable
np.random.seed(seed)     # for numpy
random.set_seed(seed)    # for tensorflow/keras
python_random.seed(seed) # for python

adam = Adam(learning_rate=0.001) # also try 0.01
latent_dim = 10
n_inputs   = 9   # number of features
n_outputs  = 9   # number of features

#--- STEP 1: Base Accuracy for Real Dataset

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
label = ['Outcome']  # OutCome column is the label (binary 0/1) 
X = data[features]
y = data[label] 

# Real data split into train/test dataset for classification with random forest

X_true_train, X_true_test, y_true_train, y_true_test = train_test_split(X, y, test_size=0.30, random_state=42)
clf_true = RandomForestClassifier(n_estimators=100)
clf_true.fit(X_true_train,y_true_train)
y_true_pred=clf_true.predict(X_true_test)
print("Base Accuracy: %5.3f" % (metrics.accuracy_score(y_true_test, y_true_pred)))
print("Base classification report:\n",metrics.classification_report(y_true_test, y_true_pred))


#--- STEP 2: Generate Synthetic Data

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
    X = data.sample(n)   # sample from real data
    y = np.ones((n, 1))  # class label = 1 for real data
    return X, y

def define_generator(latent_dim, n_outputs): 
    model = Sequential()
    model.add(Dense(15, activation='relu',  kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mean_absolute_error']) # 
    return model

def define_discriminator(n_inputs):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
    return model

def define_gan(generator, discriminator):
    discriminator.trainable = False # weights must be set to not trainable
    model = Sequential()
    model.add(generator) 
    model.add(discriminator) 
    model.compile(loss='binary_crossentropy', optimizer=adam)  
    return model

def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=200): 
    
    # determine half the size of one batch, for updating the  discriminator
    half_batch = int(n_batch / 2)
    d_history = [] 
    g_history = [] 

    for epoch in range(n_epochs+1): 
                 
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

        if epoch % n_eval == 0: # evaluate the model every n_eval epochs
            print('>%d, d1=%.3f, d2=%.3f d=%.3f g=%.3f' % (epoch, d_loss_real, d_loss_fake, d_loss,  g_loss_fake))       
            plt.subplot(1, 1, 1)
            plt.plot(d_history, label='d')
            plt.plot(g_history, label='gen')
            # plt.show() # un-comment to see the plots
            plt.close()
            
    return(g_model)

#--- main part for building & training model

discriminator = define_discriminator(n_inputs)
discriminator.summary()
generator = define_generator(latent_dim, n_outputs)
generator.summary()
gan_model = define_gan(generator, discriminator)
model = train(generator, discriminator, gan_model, latent_dim)


#--- STEP 3: Evaluate the Quality of Generated Fake Data With Model

# Get nobs_synth random vectors in latent space of dimension latent_dim 
nobs_synth = 400
latent_points = generate_latent_points(latent_dim, nobs_synth)  

# produces nobs_synth synthetized data points in feature space
X = model.predict(latent_points, verbose=0)              
data_fake = pd.DataFrame(data=X,  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
data_fake.head()

# Outcome is 0 or 1. Need map the value of the generated fake data to 0 or 1.
outcome_mean = data_fake.Outcome.mean()
data_fake['Outcome'] = data_fake['Outcome'] > outcome_mean
data_fake["Outcome"] = data_fake["Outcome"].astype(int)
data_fake.to_csv('diabetes_synthetic.csv')

# do the same feature engineering in the fake data (label is Outcome column)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
label = ['Outcome']
X_fake_created = data_fake[features]
y_fake_created = data_fake[label]

# classification of synthetic data for comparison with classification of real data done before GAN
X_fake_train, X_fake_test, y_fake_train, y_fake_test = train_test_split(X_fake_created, y_fake_created, test_size=0.30, random_state=42)
clf_fake = RandomForestClassifier(n_estimators=100)
clf_fake.fit(X_fake_train,y_fake_train)
y_fake_pred=clf_fake.predict(X_fake_test)
print("Accuracy of fake data model: %5.3f" % (metrics.accuracy_score(y_fake_test, y_fake_pred)))
print("Classification report of fake data model:\n",metrics.classification_report(y_fake_test, y_fake_pred))


#--- STEP 4: Evaluate the Quality of Generated Fake Data With Table_evaluator

from table_evaluator import load_data, TableEvaluator

table_evaluator = TableEvaluator(data, data_fake)
table_evaluator.evaluate(target_col='Outcome')
# table_evaluator.visual_evaluation() 

R_data      = np.corrcoef(data.T) # T for transpose
R_data_fake = np.corrcoef(data_fake.T)
#print(R_data)
#print(R_data_fake)
print("Avg correlation distance: %5.3f" % (np.average(abs(R_data-R_data_fake))))

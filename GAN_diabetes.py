import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random as python_random
from tensorflow import random
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import randn
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

seed = 101     # to make results reproducible
np.random.seed(seed)     # for numpy
random.set_seed(seed)    # seed for keras
python_random.seed(seed) # for python

data = pd.read_csv('diabetes.csv')
print (data.shape)
print (data.tail())
print (data.columns)

## To run reprodible runs:
## CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python GAN_diabetesb.py


#--- STEP 1: Base Accuracy for Real Dataset

# In this section we will use the real data to train a Random Forest model and get the accuracy of the model. The accuracy of the model trained from the real data is used as the base accuracy to compare with the generated fake data. 

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
label = ['Outcome']  # OutCome column is the label (binary 0/1) 
X = data[features]
y = data[label] 

# The real dataset is split into train and test dataset. The random forest classifier model is trained and evaluate the accuracy.

X_true_train, X_true_test, y_true_train, y_true_test = train_test_split(X, y, test_size=0.30, random_state=42)
clf_true = RandomForestClassifier(n_estimators=100)
clf_true.fit(X_true_train,y_true_train)
y_true_pred=clf_true.predict(X_true_test)
print("Base Accuracy: %5.3f" % (metrics.accuracy_score(y_true_test, y_true_pred)))
print("Base classification report:",metrics.classification_report(y_true_test, y_true_pred))

# We get the accuracy of the base model for real data is around 0.76; Precision is around 0.82. The accuracy of the model trained from real data will be the base accuracy to compare with the model trained from generated fake data in the further steps.


#--- STEP 2: Generate Synthetic Data

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples) 
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# We define the generate_fake_samples function to produce fake data. The input of the generator will be the created latent points (random noise). The generator will predict the input random noise and output a numpy array. Because it is the fake data, the label will be 0. 

def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples) # random N(0,1) data
    X = generator.predict(x_input,verbose=0) 
    y = np.zeros((n_samples, 1))  # class label = 0 for fake data
    return X, y

def generate_real_samples(n):
    X = data.sample(n)   # sample from real data
    y = np.ones((n, 1))  # class label = 1 for real data
    return X, y

# We will create a simple sequential model as generator with Keras module. The input dimension will be the same as the dimension of input samples. The kernel will be initialized by ‘ he_uniform ’. The model will have 3 layers, two layers will be activated by ‘relu’ function. The output layer will be activated by ‘linear’ function and the dimension of the output layer is the same as the dimension of the dataset (9 columns).

def define_generator(latent_dim, n_outputs=9):
    model = Sequential()
    model.add(Dense(15, activation='relu',  kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    # compile below is an addition from GAN_signal.py (no compile here in original version)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model

# After we have defined the generator, we will define the discriminator next step. The discriminator is also a simple sequential model including 3 dense layers. The first two layers are activated by ‘relu’ function, the output layer is activated by ‘sigmoid’ function because it will discriminate the input samples are real (True) or fake (False).
# optimizer = 'SGD'  may be more stable

def define_discriminator(n_inputs=9):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def define_gan(generator, discriminator):
    discriminator.trainable = False # weights must be set to not trainablr
    model = Sequential()
    model.add(generator) 
    model.add(discriminator) 
    model.compile(loss='binary_crossentropy', optimizer='adam')  
    return model

# Finally we will train the generator and discriminator. For each epoch, we will combine half batch of real data and half batch of fake data, then calculate the average loss. The combined model will be updated based on train_on_batch function. The trained generator will be saved for further use.

def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=500): ## 10000 epochs
    
    # determine half the size of one batch, for updating the  discriminator
    half_batch = int(n_batch / 2)
    d_history = [] 
    g_history = [] 

    for epoch in range(n_epochs):
    
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

        # update discriminator
        d_loss_real, d_real_acc = d_model.train_on_batch(x_real, y_real) 
        d_loss_fake, d_fake_acc = d_model.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        x_gan = generate_latent_points(latent_dim, n_batch)  # input for generator
        y_gan = np.ones((n_batch, 1))                        # label = 1 for fake samples
  
        # update generator via the discriminator error
        g_loss_fake = gan_model.train_on_batch(x_gan, y_gan) 
        d_history.append(d_loss)
        g_history.append(g_loss_fake)

        if epoch % n_eval == 5: # evaluate the model every n_eval epochs
            print('>%d, d1=%.3f, d2=%.3f d=%.3f g=%.3f' % (epoch+1, d_loss_real, d_loss_fake, d_loss,  g_loss_fake))       
            plt.subplot(1, 1, 1)
            plt.plot(d_history, label='d')
            plt.plot(g_history, label='gen')
            plt.show() 
            plt.close()
            
    return(g_model)

#--- main part for building & training model

latent_dim = 10
discriminator = define_discriminator()
discriminator.summary()
generator = define_generator(latent_dim)
generator.summary()
gan_model = define_gan(generator, discriminator)
model = train(generator, discriminator, gan_model, latent_dim)


#--- STEP 3: Evaluate the Quality of Generated Fake Data With Model

latent_points = generate_latent_points(10, 750)
X = model.predict(latent_points, verbose=0)      # produces the synthetized data
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

# We will train the random forest classifier model with the fake data and get the accuracy. It will be used to compare with the accuracy of the base model accuracy.

X_fake_train, X_fake_test, y_fake_train, y_fake_test = train_test_split(X_fake_created, y_fake_created, test_size=0.30, random_state=42)
clf_fake = RandomForestClassifier(n_estimators=100)
clf_fake.fit(X_fake_train,y_fake_train)
y_fake_pred=clf_fake.predict(X_fake_test)
print("Accuracy of fake data model: %5.3f" % (metrics.accuracy_score(y_fake_test, y_fake_pred)))
print("Classification report of fake data model:\n",metrics.classification_report(y_fake_test, y_fake_pred))

# The accuracy of the new trained model with generated fake data is around 0.88; Compared with the model trained with real data is around 0.75. It seems the fake data model is still skewed compared with the real data.
# Accuracy and its volatility depends on seed, classifier [random forest], n_obs, n_epochs, and metric used to measure accuracy. Does it depend on the batch split ratio and latent data?

#--- STEP 4: Evaluate the Quality of Generated Fake Data With Table_evaluator

# !pip install table_evaluator
from table_evaluator import load_data, TableEvaluator

table_evaluator = TableEvaluator(data, data_fake)
table_evaluator.evaluate(target_col='Outcome')
# table_evaluator.visual_evaluation() 

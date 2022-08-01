#Supporting Information for "Machine Learning from Heteroscedastic Data: Second Virial Coefficients of Alkane Isomers"
#Authors: Arpit Bansal, Andrew J. Schultz, David A. Kofke and Johannes Hachmann
#Journal of Physical Chemistry B (2022)

#Python code build a neural network and make predictions for B2 values
#Need descriptors from Dragon, shape descriptors and simulated values of B2 for alkanes to run the code

#importing relevant libraries
import pandas as pd
import numpy as np
from scipy import rand
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random

#Setting hyper-parameters for the neural network
ilr = 0.001 #Initial learning rate
k = 0.0001 #Decay rate
targetCor = 0.65 #Target correlation cut-off
l2_reg = 0.01 #L2 regularization parameter
MW_threshold = 100 #Molecular weight of minimum alkane isomer used in training
Epochs = 3000 
C20_test_size_ratio = 0.20
randomSeed = 3

file_name = "_C20_test_size_ratio_"+ str(C20_test_size_ratio) + "_C7toC15_TarCor"+str(targetCor)+"_DesCor_0.95_2D_ExpDecay_lr_"+str(ilr)+"_k_"+str(k)+"_4_HL_50_25_10_5_Steps_10_7_Uncertainty_with_CrossVal"+str(Epochs)

#setting seed value for reproducible results
random.seed(randomSeed)
tf.random.set_seed(randomSeed)

#Preparing dataset
#2D Descriptors
df = pd.read_csv('Upto_C15_Dragon_descriptors.txt', delimiter='\t', header=0, index_col=0)
df_C16 = pd.read_csv('C16_1000_Descriptors.txt', delimiter='\t', header=None, index_col=None)
df_C16.index = df_C16.index + 1
df_C16 = df_C16.iloc[:, 1:]
df_C17 = pd.read_csv('C17_1000_Descriptors.txt', delimiter='\t', header=None, index_col=None)
df_C17.index = df_C17.index + 1
df_C17 = df_C17.iloc[:, 1:]
df_C18 = pd.read_csv('C18_1000_Descriptors.txt', delimiter='\t', header=None, index_col=None)
df_C18.index = df_C18.index + 1
df_C18 = df_C18.iloc[:, 1:]
df_C19 = pd.read_csv('C19_1000_Descriptors.txt', delimiter='\t', header=None, index_col=None)
df_C19.index = df_C19.index + 1
df_C19 = df_C19.iloc[:, 1:]
df_C20 = pd.read_csv('C20_1000_Descriptors.txt', delimiter='\t', header=None, index_col=None)
df_C20.index = df_C20.index + 1
df_C20 = df_C20.iloc[:, 1:]

#df = df['MW']
#2D + 3D Descriptors
#df = pd.read_csv('3D_Descriptors.csv', delimiter=',', header=0, index_col=1)
#df = df.iloc[:, 2:]
#Shape Descriptors
#ds = pd.read_csv('Shape_Descriptors_300_100000.csv', delimiter=',', header=0, index_col=0)
#ds = ds.iloc[:, 1:]
#df = pd.concat([ds, df], axis = 1)
#first_column = df.pop('MW')
#df.insert(0, 'MW', first_column)

dY = pd.read_csv('B2_C15_300_10000000_sigmaHSRef_10.txt', delimiter='\t', header=None, index_col=None)
dY.index = dY.index + 1
dY = dY.iloc[:, 1:]
dY_C16 = pd.read_csv('B2_C16_300_10000000_sigmaHSRef_10.txt', delimiter='\t', header=None, index_col=None)
dY_C16.index = dY_C16.index + 1
dY_C16 = dY_C16.iloc[:, 1:]
dY_C17 = pd.read_csv('B2_C17_300_10000000_sigmaHSRef_10.txt', delimiter='\t', header=None, index_col=None)
dY_C17.index = dY_C17.index + 1
dY_C17 = dY_C17.iloc[:, 1:]
dY_C18 = pd.read_csv('B2_C18_300_10000000_sigmaHSRef_10.txt', delimiter='\t', header=None, index_col=None)
dY_C18.index = dY_C18.index + 1
dY_C18 = dY_C18.iloc[:, 1:]
dY_C19 = pd.read_csv('B2_C19_300_10000000_sigmaHSRef_10.txt', delimiter='\t', header=None, index_col=None)
dY_C19.index = dY_C19.index + 1
dY_C19 = dY_C19.iloc[:, 1:]
dY_C20 = pd.read_csv('B2_C20_300_10000000_sigmaHSRef_10.txt', delimiter='\t', header=None, index_col=None)
dY_C20.index = dY_C20.index + 1
dY_C20 = dY_C20.iloc[:, 1:]

df = pd.concat([df, dY], axis = 1)
df=df.rename(columns={1: "B2"})
df=df.rename(columns={2: "uncertainty"})

df_C16 = pd.concat([df_C16, dY_C16], axis = 1, ignore_index = True)
df_C17 = pd.concat([df_C17, dY_C17], axis = 1, ignore_index = True)
df_C18 = pd.concat([df_C18, dY_C18], axis = 1, ignore_index = True)
df_C19 = pd.concat([df_C19, dY_C19], axis = 1, ignore_index = True)
df_C20 = pd.concat([df_C20, dY_C20], axis = 1, ignore_index = True)

df_C16.columns = df.columns
df_C17.columns = df.columns
df_C18.columns = df.columns
df_C19.columns = df.columns
df_C20.columns = df.columns

# Removing methane as it has a lot of missing values
df = df.drop(df[df['NAME'] == "C"].index)

df = df.apply(pd.to_numeric, errors='coerce')

df=df.replace(to_replace='nan', value = np.nan)
df=df.replace(to_replace='na', value = np.nan)

#remove features with missing values
df=df.drop(df.isna().sum().sort_values()[df.isna().sum().sort_values()>0].index, axis=1)

#remove features with all zeros
df=df.loc[:, (df != 0).any(axis=0)]

#remove features with same values for all molecules
nunique = df.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
df=df.drop(cols_to_drop, axis=1)

#Feature selection
corr_matrix = df.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns[:-2] if any(upper[column] > 0.95)]
# Drop features 
df.drop(to_drop, axis=1, inplace=True)

corr_matrix = df.corr().abs()
cor_target = corr_matrix["B2"]
# Select highly correlated features (thresold = 0.65)
relevant_features = cor_target[cor_target>targetCor]
names = [index for index, value in relevant_features.iteritems()]
if 'MW' not in names:
    names = ['MW'] + names
df = df[names] 

df_C16 = df_C16[df.columns]
df_C17 = df_C17[df.columns]
df_C18 = df_C18[df.columns]
df_C19 = df_C19[df.columns]
df_C20 = df_C20[df.columns]

df_C16=df_C16.replace(to_replace='nan', value = np.nan)
df_C16=df_C16.replace(to_replace='na', value = np.nan)
df_C16=df_C16.drop(df_C16.isna().sum(axis=1).sort_values()[df_C16.isna().sum(axis=1).sort_values()>0].index, axis=0)

df_C17=df_C17.replace(to_replace='nan', value = np.nan)
df_C17=df_C17.replace(to_replace='na', value = np.nan)
df_C17=df_C17.drop(df_C17.isna().sum(axis=1).sort_values()[df_C17.isna().sum(axis=1).sort_values()>0].index, axis=0)

df_C18=df_C18.replace(to_replace='nan', value = np.nan)
df_C18=df_C18.replace(to_replace='na', value = np.nan)
df_C18=df_C18.drop(df_C18.isna().sum(axis=1).sort_values()[df_C18.isna().sum(axis=1).sort_values()>0].index, axis=0)

df_C19=df_C19.replace(to_replace='nan', value = np.nan)
df_C19=df_C19.replace(to_replace='na', value = np.nan)
df_C19=df_C19.drop(df_C19.isna().sum(axis=1).sort_values()[df_C19.isna().sum(axis=1).sort_values()>0].index, axis=0)

df_C20=df_C20.replace(to_replace='nan', value = np.nan)
df_C20=df_C20.replace(to_replace='na', value = np.nan)
df_C20=df_C20.drop(df_C20.isna().sum(axis=1).sort_values()[df_C20.isna().sum(axis=1).sort_values()>0].index, axis=0)

X = df.iloc[:,:-2].values
y = df.iloc[:, -2:].values
y = y.reshape(len(y),2)

x_C16 = df_C16.iloc[:,:-2].values
y_C16 = df_C16.iloc[:, -2:].values
y_C16 = y_C16.reshape(len(y_C16),2)

x_C17 = df_C17.iloc[:,:-2].values
y_C17 = df_C17.iloc[:, -2:].values
y_C17 = y_C17.reshape(len(y_C17),2)

x_C18 = df_C18.iloc[:,:-2].values
y_C18 = df_C18.iloc[:, -2:].values
y_C18 = y_C18.reshape(len(y_C18),2)

x_C19 = df_C19.iloc[:,:-2].values
y_C19 = df_C19.iloc[:, -2:].values
y_C19 = y_C19.reshape(len(y_C19),2)

x_C20 = df_C20.iloc[:,:-2].values
y_C20 = df_C20.iloc[:, -2:].values
y_C20 = y_C20.reshape(len(y_C20),2)

from sklearn.model_selection import train_test_split
y_C15 = y[np.logical_and(X[:, 0]>MW_threshold, X[:, 0]<215)]
x_C15 = X[np.logical_and(X[:, 0]>MW_threshold, X[:, 0]<215)]

from sklearn.utils import shuffle
x_C15, y_C15 = shuffle(x_C15, y_C15, random_state = randomSeed)
x_C16, y_C16 = shuffle(x_C16, y_C16, random_state = randomSeed)
x_C17, y_C17 = shuffle(x_C17, y_C17, random_state = randomSeed)
x_C18, y_C18 = shuffle(x_C18, y_C18, random_state = randomSeed)
x_C19, y_C19 = shuffle(x_C19, y_C19, random_state = randomSeed)
x_C20, y_C20 = shuffle(x_C20, y_C20, random_state = randomSeed)

# Keeping 20% C15 as test set, independent of C15_test_size for like to like comparison across different C15_test_size models
x_train_val, x_test, Y_train_val, y_test = train_test_split(x_C15, y_C15, test_size = 0.2, random_state = randomSeed)
x_train_val_test_C16, x_test_C16, Y_train_val_test_C16, y_test_C16 = train_test_split(x_C16, y_C16, test_size = 0.2, random_state = randomSeed)
x_train_val_test_C17, x_test_C17, Y_train_val_test_C17, y_test_C17 = train_test_split(x_C17, y_C17, test_size = 0.2, random_state = randomSeed)
x_train_val_test_C18, x_test_C18, Y_train_val_test_C18, y_test_C18 = train_test_split(x_C18, y_C18, test_size = 0.2, random_state = randomSeed)
x_train_val_test_C19, x_test_C19, Y_train_val_test_C19, y_test_C19 = train_test_split(x_C19, y_C19, test_size = 0.2, random_state = randomSeed)
x_train_val_test_C20, x_test_C20, Y_train_val_test_C20, y_test_C20 = train_test_split(x_C20, y_C20, test_size = 0.2, random_state = randomSeed)

if(C20_test_size_ratio == 1.0):
    x_train_val = x_train_val
    Y_train_val = Y_train_val
elif(C20_test_size_ratio == 0.2):
    x_train_val_C16 = x_train_val_test_C16
    Y_train_val_C16 = Y_train_val_test_C16
    x_train_val_C17 = x_train_val_test_C17
    Y_train_val_C17 = Y_train_val_test_C17
    x_train_val_C18 = x_train_val_test_C18
    Y_train_val_C18 = Y_train_val_test_C18
    x_train_val_C19 = x_train_val_test_C19
    Y_train_val_C19 = Y_train_val_test_C19
    x_train_val_C20 = x_train_val_test_C20
    Y_train_val_C20 = Y_train_val_test_C20
    x_train_val = np.concatenate((x_train_val, x_train_val_C16, x_train_val_C17, x_train_val_C18, x_train_val_C19, x_train_val_C20), axis=0)
    Y_train_val = np.concatenate((Y_train_val, Y_train_val_C16, Y_train_val_C17, Y_train_val_C18, Y_train_val_C19, Y_train_val_C20), axis=0)    
else:
    x_train_val_test_C16, Y_train_val_test_C16 = shuffle(x_train_val_test_C16, Y_train_val_test_C16, random_state = randomSeed)
    x_train_val_C16, x_test_remC16, Y_train_val_C16, y_test_remC16 = train_test_split(x_train_val_test_C16, Y_train_val_test_C16, test_size = (C20_test_size_ratio - 0.2)/0.8, random_state = randomSeed)
    x_train_val_test_C17, Y_train_val_test_C17 = shuffle(x_train_val_test_C17, Y_train_val_test_C17, random_state = randomSeed)
    x_train_val_C17, x_test_remC17, Y_train_val_C17, y_test_remC17 = train_test_split(x_train_val_test_C17, Y_train_val_test_C17, test_size = (C20_test_size_ratio - 0.2)/0.8, random_state = randomSeed)
    x_train_val_test_C18, Y_train_val_test_C18 = shuffle(x_train_val_test_C18, Y_train_val_test_C18, random_state = randomSeed)
    x_train_val_C18, x_test_remC18, Y_train_val_C18, y_test_remC18 = train_test_split(x_train_val_test_C18, Y_train_val_test_C18, test_size = (C20_test_size_ratio - 0.2)/0.8, random_state = randomSeed)
    x_train_val_test_C19, Y_train_val_test_C19 = shuffle(x_train_val_test_C19, Y_train_val_test_C19, random_state = randomSeed)
    x_train_val_C19, x_test_remC19, Y_train_val_C19, y_test_remC19 = train_test_split(x_train_val_test_C19, Y_train_val_test_C19, test_size = (C20_test_size_ratio - 0.2)/0.8, random_state = randomSeed)
    x_train_val_test_C20, Y_train_val_test_C20 = shuffle(x_train_val_test_C20, Y_train_val_test_C20, random_state = randomSeed)
    x_train_val_C20, x_test_remC20, Y_train_val_C20, y_test_remC20 = train_test_split(x_train_val_test_C20, Y_train_val_test_C20, test_size = (C20_test_size_ratio - 0.2)/0.8, random_state = randomSeed)
    x_train_val = np.concatenate((x_train_val, x_train_val_C16, x_train_val_C17, x_train_val_C18, x_train_val_C19, x_train_val_C20), axis=0)
    Y_train_val = np.concatenate((Y_train_val, Y_train_val_C16, Y_train_val_C17, Y_train_val_C18, Y_train_val_C19, Y_train_val_C20), axis=0)

y_test_C15 = y_test
x_test_C15 = x_test

from sklearn.utils import shuffle
x_train_val, Y_train_val = shuffle(x_train_val, Y_train_val, random_state = randomSeed)

uncertainty_test = y_test[:, 1]
uncertainty_test = uncertainty_test.reshape(len(uncertainty_test),1)

uncertainty_test_C15 = y_test_C15[:, 1]
uncertainty_test_C15 = uncertainty_test_C15.reshape(len(uncertainty_test_C15),1)

uncertainty_test_C16 = y_test_C16[:, 1]
uncertainty_test_C16 = uncertainty_test_C16.reshape(len(uncertainty_test_C16),1)

uncertainty_test_C17 = y_test_C17[:, 1]
uncertainty_test_C17 = uncertainty_test_C17.reshape(len(uncertainty_test_C17),1)

uncertainty_test_C18 = y_test_C18[:, 1]
uncertainty_test_C18 = uncertainty_test_C18.reshape(len(uncertainty_test_C18),1)

uncertainty_test_C19 = y_test_C19[:, 1]
uncertainty_test_C19 = uncertainty_test_C19.reshape(len(uncertainty_test_C19),1)

uncertainty_test_C20 = y_test_C20[:, 1]
uncertainty_test_C20 = uncertainty_test_C20.reshape(len(uncertainty_test_C20),1)

# Creating 5 folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
f = open("L2_"+str(l2_reg)+str(file_name)+".txt", "w")
 
Train_index = []
Val_index = []
for train_index, val_index in kf.split(x_train_val, Y_train_val):
    train_index = train_index.reshape(len(train_index), 1)
    Train_index.append(train_index)
    val_index = val_index.reshape(len(val_index), 1)
    Val_index.append(val_index)

#Building model using each of the 5 folds and making predictions
for j in range(5):
    train_index = Train_index[j]
    val_index = Val_index[j]
    print("TRAIN:", train_index, "VAL:", val_index)
    X_train, X_val = x_train_val[train_index], x_train_val[val_index]
    y_train, y_val = Y_train_val[train_index], Y_train_val[val_index]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[2])
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[2])
    
    X_train, y_train = shuffle(X_train, y_train, random_state=randomSeed)
    
    X_test = x_test
    X_test_C15 = x_test_C15
    X_test_C16 = x_test_C16
    X_test_C17 = x_test_C17
    X_test_C18 = x_test_C18
    X_test_C19 = x_test_C19
    X_test_C20 = x_test_C20
    x_train = X_train
    x_val = X_val

    # Exponentially decaying learning rate
    def lr_exp_decay(epoch, lr):
        return ilr * math.exp(-k*epoch)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)
    X_test_C15 = sc.transform(X_test_C15)
    X_test_C16 = sc.transform(X_test_C16)
    X_test_C17 = sc.transform(X_test_C17)
    X_test_C18 = sc.transform(X_test_C18)
    X_test_C19 = sc.transform(X_test_C19)
    X_test_C20 = sc.transform(X_test_C20)
    
    # Part 2 - Building the ANN
    
    # Initializing the ANN
    ann = tf.keras.models.Sequential()
    
    initializer = tf.keras.initializers.HeNormal(seed=randomSeed)
    
    # Adding the seventh hidden layer
    #ann.add(tf.keras.layers.Dense(units=500, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(
    #    l2=l2_reg), kernel_initializer=initializer))

    # Adding the sixth hidden layer
    #ann.add(tf.keras.layers.Dense(units=200, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(
    #    l2=l2_reg), kernel_initializer=initializer))

    # Adding the fifth hidden layer
    #ann.add(tf.keras.layers.Dense(units=100, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(
    #    l2=l2_reg), kernel_initializer=initializer))

    # Adding the first hidden layer
    ann.add(tf.keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(
        l2=l2_reg), kernel_initializer=initializer))
    
    # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=25, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(
        l2=l2_reg), kernel_initializer=initializer))

    # Adding the third hidden layer
    ann.add(tf.keras.layers.Dense(units=10, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(
        l2=l2_reg), kernel_initializer=initializer))

    # Adding the fourth hidden layer
    ann.add(tf.keras.layers.Dense(units=5, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(
        l2=l2_reg), kernel_initializer=initializer))

    # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=1))
    
    # Part 3 - Defining the loss
    def loss(y_true, y_predicted):
        return tf.keras.backend.mean(tf.keras.backend.square((y_true[:, 0] - y_predicted[:, 0])/(y_true[:, 1])))

    def MAEloss(y_true, y_predicted):
        return tf.keras.backend.mean(tf.keras.backend.abs((y_true[:, 0] - y_predicted[:, 0])/(y_true[:, 1])))

    def modifiedRMSE(y_true, y_predicted):
        return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square((y_true[:, 0] - y_predicted[:, 0])/(y_true[:, 1]))))

    # Compiling the ANN
    ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ilr), loss=loss, metrics=[modifiedRMSE])
    history = ann.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size = 32, epochs=Epochs, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1)])
    
    #ann.save("L2_"+str(l2_reg)+str(file_name)+"_"+str(j))
    uncertainty_train = y_train[:, 1]
    uncertainty_train = uncertainty_train.reshape(len(uncertainty_train),1)
    uncertainty_val = y_val[:, 1]
    uncertainty_val = uncertainty_val.reshape(len(uncertainty_val),1)

    p = open("Training_Loss_RMSE_LR_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("Epochs \t LR \t TrainLoss \t ValLoss \t TrainModifiedRMSE \t ValModifiedRMSE\n")
    for i in range(Epochs):
        p.write(str(i+1) + "\t" + str(history.history['lr'][i]) + "\t" + str(history.history['loss'][i]) + "\t" +str(history.history['val_loss'][i]) + "\t" +str(history.history['modifiedRMSE'][i]) + "\t" +str(history.history['val_modifiedRMSE'][i]) + "\n")
    p.close()
    
    y_train = y_train[:, 0]
    y_val = y_val[:, 0]
    y_test = y_test[:, 0]
    y_test_C15 = y_test_C15[:, 0]
    y_test_C16 = y_test_C16[:, 0]
    y_test_C17 = y_test_C17[:, 0]
    y_test_C18 = y_test_C18[:, 0]
    y_test_C19 = y_test_C19[:, 0]
    y_test_C20 = y_test_C20[:, 0]

    y_train = y_train.reshape(len(y_train),1)
    y_val = y_val.reshape(len(y_val),1)
    y_test = y_test.reshape(len(y_test),1)
    y_test_C15 = y_test_C15.reshape(len(y_test_C15),1)
    y_test_C16 = y_test_C16.reshape(len(y_test_C16),1)
    y_test_C17 = y_test_C17.reshape(len(y_test_C17),1)
    y_test_C18 = y_test_C18.reshape(len(y_test_C18),1)
    y_test_C19 = y_test_C19.reshape(len(y_test_C19),1)
    y_test_C20 = y_test_C20.reshape(len(y_test_C20),1)

    #Using the trained network to make predictions
    y_pred_train = ann.predict(X_train)
    y_pred_val = ann.predict(X_val)
    y_pred = ann.predict(X_test)
    y_pred_C15 = ann.predict(X_test_C15)
    y_pred_C16 = ann.predict(X_test_C16)
    y_pred_C17 = ann.predict(X_test_C17)
    y_pred_C18 = ann.predict(X_test_C18)
    y_pred_C19 = ann.predict(X_test_C19)
    y_pred_C20 = ann.predict(X_test_C20)
    
    pe = 100*(y_test - y_pred)/y_test
    pe_train = 100*(y_train - y_pred_train)/y_train
    pe_val = 100*(y_val - y_pred_val)/y_val
    pe_C15 = 100*(y_test_C15 - y_pred_C15)/y_test_C15
    pe_C16 = 100*(y_test_C16 - y_pred_C16)/y_test_C16
    pe_C17 = 100*(y_test_C17 - y_pred_C17)/y_test_C17
    pe_C18 = 100*(y_test_C18 - y_pred_C18)/y_test_C18
    pe_C19 = 100*(y_test_C19 - y_pred_C19)/y_test_C19
    pe_C20 = 100*(y_test_C20 - y_pred_C20)/y_test_C20
    
    #Saving predictions
    p = open("Training Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("Training Prediction \n")
    for i in range(len(y_train)):
        p.write(str(y_train[i])[1:-1] + "\t" + str(y_pred_train[i])[1:-1] + "\t" + str(uncertainty_train[i])[1:-1] + "\n")
    p.close()
    p = open("Validation Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("Validation Prediction \n")
    for i in range(len(y_val)):
        p.write(str(y_val[i])[1:-1] + "\t" + str(y_pred_val[i])[1:-1] + "\t" + str(uncertainty_val[i])[1:-1] + "\n")
    p.close()
    p = open("Test Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("Test Prediction \n")
    for i in range(len(y_test)):
        p.write(str(y_test[i])[1:-1] + "\t" + str(y_pred[i])[1:-1] + "\t" + str(uncertainty_test[i])[1:-1] + "\n")
    p.close()
    p = open("C15 Test Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("C15 Test Prediction \n")
    for i in range(len(y_test_C15)):
        p.write(str(y_test_C15[i])[1:-1] + "\t" + str(y_pred_C15[i])[1:-1] + "\t" + str(uncertainty_test_C15[i])[1:-1] + "\n")
    p.close()
    
    p = open("C16 Test Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("C16 Test Prediction \n")
    for i in range(len(y_test_C16)):
        p.write(str(y_test_C16[i])[1:-1] + "\t" + str(y_pred_C16[i])[1:-1] + "\t" + str(uncertainty_test_C16[i])[1:-1] + "\n")
    p.close()
    
    p = open("C17 Test Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("C17 Test Prediction \n")
    for i in range(len(y_test_C17)):
        p.write(str(y_test_C17[i])[1:-1] + "\t" + str(y_pred_C17[i])[1:-1] + "\t" + str(uncertainty_test_C17[i])[1:-1] + "\n")
    p.close()
    
    p = open("C18 Test Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("C18 Test Prediction \n")
    for i in range(len(y_test_C18)):
        p.write(str(y_test_C18[i])[1:-1] + "\t" + str(y_pred_C18[i])[1:-1] + "\t" + str(uncertainty_test_C18[i])[1:-1] + "\n")
    p.close()
    
    p = open("C19 Test Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("C19 Test Prediction \n")
    for i in range(len(y_test_C19)):
        p.write(str(y_test_C19[i])[1:-1] + "\t" + str(y_pred_C19[i])[1:-1] + "\t" + str(uncertainty_test_C19[i])[1:-1] + "\n")
    p.close()
    
    p = open("C20 Test Predictions_L2_"+str(l2_reg)+str(file_name)+"_"+str(j)+".txt", "w")
    p.write("C20 Test Prediction \n")
    for i in range(len(y_test_C20)):
        p.write(str(y_test_C20[i])[1:-1] + "\t" + str(y_pred_C20[i])[1:-1] + "\t" + str(uncertainty_test_C20[i])[1:-1] + "\n")
    p.close()

    #Computing relevant metrics to track the performance of the neural network
    
    #print("Training set results \n")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_train, y_pred_train)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_train, y_pred_train)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    mape = np.mean(np.abs(100*(y_train - y_pred_train) / y_train))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_train - y_pred_train) / y_train)))

    me = np.mean(y_train - y_pred_train)

    mpe = np.mean(100*(y_train - y_pred_train) / y_train)

    maxae = np.amax(np.abs(y_train - y_pred_train))

    maxape = np.amax(np.abs(100*(y_train - y_pred_train) / y_train))

    deltamaxe = np.amax(y_train - y_pred_train) - np.amin(y_train - y_pred_train) 

    r2_train = r2_score(y_train/uncertainty_train, y_pred_train/uncertainty_train)

    mae_train = mean_absolute_error(y_train/uncertainty_train, y_pred_train/uncertainty_train)

    rmse_train = np.sqrt(mean_squared_error(y_train/uncertainty_train, y_pred_train/uncertainty_train))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe)  + "\t" + str(r2_train) + "\t" + str(mae_train) + "\t"+ str(rmse_train)  + "\n")#, sep = '\t')

    #print("Validation set results \n")

    from sklearn.metrics import r2_score
    r2 = r2_score(y_val, y_pred_val)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_val, y_pred_val)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    mape = np.mean(np.abs(100*(y_val - y_pred_val) / y_val))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_val - y_pred_val) / y_val)))

    me = np.mean(y_val - y_pred_val)

    mpe = np.mean(100*(y_val - y_pred_val) / y_val)

    maxae = np.amax(np.abs(y_val - y_pred_val))

    maxape = np.amax(np.abs(100*(y_val - y_pred_val) / y_val))

    deltamaxe = np.amax(y_val - y_pred_val) - np.amin(y_val - y_pred_val)

    r2_val = r2_score(y_val/uncertainty_val, y_pred_val/uncertainty_val)
    
    mae_val = mean_absolute_error(y_val/uncertainty_val, y_pred_val/uncertainty_val)
    
    rmse_val = np.sqrt(mean_squared_error(y_val/uncertainty_val, y_pred_val/uncertainty_val))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe)  + "\t" + str(r2_val) + "\t" + str(mae_val) + "\t"+ str(rmse_val) + "\n")#, sep = '\t')

    #print("Test set results \n")

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test, y_pred)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mape = np.mean(np.abs(100*(y_test - y_pred) / y_test))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_test - y_pred) / y_test)))

    me = np.mean(y_test - y_pred)

    mpe = np.mean(100*(y_test - y_pred) / y_test)

    maxae = np.amax(np.abs(y_test - y_pred))

    maxape = np.amax(np.abs(100*(y_test - y_pred) / y_test))

    deltamaxe = np.amax(y_test - y_pred) - np.amin(y_test - y_pred) 

    r2_test = r2_score(y_test/uncertainty_test, y_pred/uncertainty_test)
    
    mae_test = mean_absolute_error(y_test/uncertainty_test, y_pred/uncertainty_test)
    
    rmse_test = np.sqrt(mean_squared_error(y_test/uncertainty_test, y_pred/uncertainty_test))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe)  + "\t" + str(r2_test) + "\t" + str(mae_test) + "\t"+ str(rmse_test) + "\n")#, sep = '\t')

    #print("Test set results for only C15\n")

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_C15, y_pred_C15)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_C15, y_pred_C15)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test_C15, y_pred_C15))

    mape = np.mean(np.abs(100*(y_test_C15 - y_pred_C15) / y_test_C15))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_test_C15 - y_pred_C15) / y_test_C15)))

    me = np.mean(y_test_C15 - y_pred_C15)

    mpe = np.mean(100*(y_test_C15 - y_pred_C15) / y_test_C15)

    maxae = np.amax(np.abs(y_test_C15 - y_pred_C15))

    maxape = np.amax(np.abs(100*(y_test_C15 - y_pred_C15) / y_test_C15))

    deltamaxe = np.amax(y_test_C15 - y_pred_C15) - np.amin(y_test_C15 - y_pred_C15) 

    r2_test_C15 = r2_score(y_test_C15/uncertainty_test_C15, y_pred_C15/uncertainty_test_C15)

    mae_test_C15 = mean_absolute_error(y_test_C15/uncertainty_test_C15, y_pred_C15/uncertainty_test_C15)
    
    rmse_test_C15 = np.sqrt(mean_squared_error(y_test_C15/uncertainty_test_C15, y_pred_C15/uncertainty_test_C15))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe) + "\t" + str(r2_test_C15) + "\t" + str(mae_test_C15) + "\t" + str(rmse_test_C15) + "\n")#, sep = '\t')

    #print("Test set results for only C16\n")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_C16, y_pred_C16)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_C16, y_pred_C16)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test_C16, y_pred_C16))

    mape = np.mean(np.abs(100*(y_test_C16 - y_pred_C16) / y_test_C16))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_test_C16 - y_pred_C16) / y_test_C16)))

    me = np.mean(y_test_C16 - y_pred_C16)

    mpe = np.mean(100*(y_test_C16 - y_pred_C16) / y_test_C16)

    maxae = np.amax(np.abs(y_test_C16 - y_pred_C16))

    maxape = np.amax(np.abs(100*(y_test_C16 - y_pred_C16) / y_test_C16))

    deltamaxe = np.amax(y_test_C16 - y_pred_C16) - np.amin(y_test_C16 - y_pred_C16) 

    r2_test_C16 = r2_score(y_test_C16/uncertainty_test_C16, y_pred_C16/uncertainty_test_C16)

    mae_test_C16 = mean_absolute_error(y_test_C16/uncertainty_test_C16, y_pred_C16/uncertainty_test_C16)
    
    rmse_test_C16 = np.sqrt(mean_squared_error(y_test_C16/uncertainty_test_C16, y_pred_C16/uncertainty_test_C16))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe) + "\t" + str(r2_test_C16) + "\t" + str(mae_test_C16) + "\t" + str(rmse_test_C16) + "\n")#, sep = '\t')

    #print("Test set results for only C17\n")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_C17, y_pred_C17)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_C17, y_pred_C17)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test_C17, y_pred_C17))

    mape = np.mean(np.abs(100*(y_test_C17 - y_pred_C17) / y_test_C17))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_test_C17 - y_pred_C17) / y_test_C17)))

    me = np.mean(y_test_C17 - y_pred_C17)

    mpe = np.mean(100*(y_test_C17 - y_pred_C17) / y_test_C17)

    maxae = np.amax(np.abs(y_test_C17 - y_pred_C17))

    maxape = np.amax(np.abs(100*(y_test_C17 - y_pred_C17) / y_test_C17))

    deltamaxe = np.amax(y_test_C17 - y_pred_C17) - np.amin(y_test_C17 - y_pred_C17) 

    r2_test_C17 = r2_score(y_test_C17/uncertainty_test_C17, y_pred_C17/uncertainty_test_C17)

    mae_test_C17 = mean_absolute_error(y_test_C17/uncertainty_test_C17, y_pred_C17/uncertainty_test_C17)
    
    rmse_test_C17 = np.sqrt(mean_squared_error(y_test_C17/uncertainty_test_C17, y_pred_C17/uncertainty_test_C17))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe) + "\t" + str(r2_test_C17) + "\t" + str(mae_test_C17) + "\t" + str(rmse_test_C17) + "\n")#, sep = '\t')

    #print("Test set results for only C18\n")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_C18, y_pred_C18)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_C18, y_pred_C18)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test_C18, y_pred_C18))

    mape = np.mean(np.abs(100*(y_test_C18 - y_pred_C18) / y_test_C18))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_test_C18 - y_pred_C18) / y_test_C18)))

    me = np.mean(y_test_C18 - y_pred_C18)

    mpe = np.mean(100*(y_test_C18 - y_pred_C18) / y_test_C18)

    maxae = np.amax(np.abs(y_test_C18 - y_pred_C18))

    maxape = np.amax(np.abs(100*(y_test_C18 - y_pred_C18) / y_test_C18))

    deltamaxe = np.amax(y_test_C18 - y_pred_C18) - np.amin(y_test_C18 - y_pred_C18) 

    r2_test_C18 = r2_score(y_test_C18/uncertainty_test_C18, y_pred_C18/uncertainty_test_C18)

    mae_test_C18 = mean_absolute_error(y_test_C18/uncertainty_test_C18, y_pred_C18/uncertainty_test_C18)
    
    rmse_test_C18 = np.sqrt(mean_squared_error(y_test_C18/uncertainty_test_C18, y_pred_C18/uncertainty_test_C18))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe) + "\t" + str(r2_test_C18) + "\t" + str(mae_test_C18) + "\t" + str(rmse_test_C18) + "\n")#, sep = '\t')

    #print("Test set results for only C19\n")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_C19, y_pred_C19)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_C19, y_pred_C19)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test_C19, y_pred_C19))

    mape = np.mean(np.abs(100*(y_test_C19 - y_pred_C19) / y_test_C19))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_test_C19 - y_pred_C19) / y_test_C19)))

    me = np.mean(y_test_C19 - y_pred_C19)

    mpe = np.mean(100*(y_test_C19 - y_pred_C19) / y_test_C19)

    maxae = np.amax(np.abs(y_test_C19 - y_pred_C19))

    maxape = np.amax(np.abs(100*(y_test_C19 - y_pred_C19) / y_test_C19))

    deltamaxe = np.amax(y_test_C19 - y_pred_C19) - np.amin(y_test_C19 - y_pred_C19) 

    r2_test_C19 = r2_score(y_test_C19/uncertainty_test_C19, y_pred_C19/uncertainty_test_C19)

    mae_test_C19 = mean_absolute_error(y_test_C19/uncertainty_test_C19, y_pred_C19/uncertainty_test_C19)
    
    rmse_test_C19 = np.sqrt(mean_squared_error(y_test_C19/uncertainty_test_C19, y_pred_C19/uncertainty_test_C19))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe) + "\t" + str(r2_test_C19) + "\t" + str(mae_test_C19) + "\t" + str(rmse_test_C19) + "\n")#, sep = '\t')

    #print("Test set results for only C20\n")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_C20, y_pred_C20)
    
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_C20, y_pred_C20)

    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test_C20, y_pred_C20))

    mape = np.mean(np.abs(100*(y_test_C20 - y_pred_C20) / y_test_C20))
    
    rmspe = np.sqrt(np.mean(np.square(100*(y_test_C20 - y_pred_C20) / y_test_C20)))

    me = np.mean(y_test_C20 - y_pred_C20)

    mpe = np.mean(100*(y_test_C20 - y_pred_C20) / y_test_C20)

    maxae = np.amax(np.abs(y_test_C20 - y_pred_C20))

    maxape = np.amax(np.abs(100*(y_test_C20 - y_pred_C20) / y_test_C20))

    deltamaxe = np.amax(y_test_C20 - y_pred_C20) - np.amin(y_test_C20 - y_pred_C20) 

    r2_test_C20 = r2_score(y_test_C20/uncertainty_test_C20, y_pred_C20/uncertainty_test_C20)

    mae_test_C20 = mean_absolute_error(y_test_C20/uncertainty_test_C20, y_pred_C20/uncertainty_test_C20)
    
    rmse_test_C20 = np.sqrt(mean_squared_error(y_test_C20/uncertainty_test_C20, y_pred_C20/uncertainty_test_C20))

    f.write(str(r2) + "\t" + str(mae)  + "\t" + str(rmse) + "\t" + str(mape) + "\t" + str(rmspe) + "\t" + str(me) + "\t" + str(mpe) + "\t" + str(maxae) + "\t" + str(maxape) + "\t" + str(deltamaxe) + "\t" + str(r2_test_C20) + "\t" + str(mae_test_C20) + "\t" + str(rmse_test_C20) + "\n")#, sep = '\t')

    f.write("\n")
f.close()

# %%

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 20:09:06 2022

@author: zhye
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import datetime as datetime
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
#import neurokit2 as nk

df = pd.read_csv('CGM2.csv')
#sub = np.unique(df['copsacno'])
indexes = np.unique(df['copsacno'], return_index=True)[1]
sub = [df['copsacno'][index] for index in sorted(indexes)]

glu_num = []
for i in range(len(sub)):
    glu_num.append(df[df['copsacno'] == sub[i]].shape[0])
    
glu_num_uni = np.unique(glu_num, return_counts=True)

plt.figure(figsize=(15,5))
plt.ylabel("Number of subjects")
plt.xlabel("Number of glucose measurements")
arr = plt.hist(glu_num, bins = 10)
for i in range(10):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
-------------------------------------------------------------------------
df_13d_index = [idx for idx, element in enumerate(glu_num) if (element > 1152) & (element < 1750)]

def TR(x, sd = 1, sr=15):
    up = np.mean(x['glucose_mmol_L']) + sd*np.std(x['glucose_mmol_L'])
    dw = np.mean(x['glucose_mmol_L']) - sd*np.std(x['glucose_mmol_L'])
    TIR = len(x[(x['glucose_mmol_L']<= up) & (x['glucose_mmol_L']>= dw)])/96
    TBR = len(x[x['glucose_mmol_L'] < dw])/96
    TAR = len(x[x['glucose_mmol_L'] > up])/96 
    return TIR, TBR, TAR

def SDRC(x):
    SDRC = np.abs(x['glucose_mmol_L'].diff()/15)
    return np.std(SDRC)

def GMI(x):
    GMI = 12.71 + (4.70587*np.mean(x['glucose_mmol_L']))
    return GMI    

def Jindex(x):
    J = 0.324*((np.mean(x['glucose_mmol_L'])+np.std(x['glucose_mmol_L']))**2)
    return J

def after_eating(x):
    AE = []
    for j in range(1, 87):
        if (x['TimeSinceLastPicture'].loc[j] < x['TimeSinceLastPicture'].loc[j - 1]):
            print(j)
            AE_temp = []
            for l in range(8):
                AE_temp.append(x['glucose_mmol_L'].loc[j + l])
            AE.append(np.mean(AE_temp))
    return np.mean(AE)

df_new = pd.DataFrame(0, index = np.arange(len(df_13d_index)), columns = ['id','T1', 'T2', 'T3', 'MG', 'SDRC', 'GMI', 'JIndex', 'after_eating'])
df_new['id'] = [sub[index] for index in df_13d_index]
for q in range(len(df_13d_index)):
#for q in range(10):
    T1 = []
    T2 = []
    T3 = []
    MG = []
    Src = []
    gmii = []
    ji = []
    af_e = []
    for t in range(0, 1151, 96):
        df_temp = df[df['copsacno'] == sub[df_13d_index[q]]].reset_index(drop = True).loc[t:t+95].reset_index(drop = True)
        T1.append(TR(df_temp)[0])
        T2.append(TR(df_temp)[1])
        T3.append(TR(df_temp)[2])
        MG.append(np.mean(df_temp['glucose_mmol_L']))
        Src.append(SDRC(df_temp))
        gmii.append(GMI(df_temp))
        ji.append(Jindex(df_temp))
        af_e.append(after_eating(df_temp))  
    #print(T1)
    #print(T2)
    #print(T3)
    #print(MG)
    #print(Src)
    #print(gmii)
    #print(ji)
    #print(af_e)
    df_new['T1'].loc[q] = np.mean(T1)
    df_new['T2'].loc[q] = np.mean(T2)
    df_new['T3'].loc[q] = np.mean(T3)
    df_new['MG'].loc[q] = np.mean(MG)
    df_new['SDRC'].loc[q] = np.mean(Src)
    df_new['GMI'].loc[q] = np.mean(gmii)
    df_new['JIndex'].loc[q] = np.mean(ji)
    df_new['after_eating'].loc[q] = np.nanmean(af_e)

df_new_noNAN = df_new.fillna(method="ffill") 
df_new_noNAN = df_new_noNAN.drop(columns='MG')

scaler = StandardScaler()
scaler.fit(np.array(df_new_noNAN[df_new_noNAN.columns[1:]]))
X = scaler.transform(np.array(df_new_noNAN[df_new_noNAN.columns[1:]]))  
pca = PCA() # estimate only 2 PCs
X_new = pca.fit_transform(X)  
      
X = preprocessing.scale(df_new_noNAN[df_new_noNAN.columns[1:]])
pca = PCA(n_components = 'mle')
pca = PCA(n_components = 0.90)
X_new = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
len(pca.explained_variance_ratio_)

def plotCumSumVariance(var=None):
    #PLOT FIGURE
    #You can use plot_color[] to obtain different colors for your plots
    #Save file
    cumvar = var.cumsum()

    plt.figure(figsize=(10,6))
    plt.rcParams.update({'font.size': 15})
    plt.bar(['PC1', 'PC1 2', 'PC1 2 3'], cumvar, width = 1.0)
    plt.axhline(y=0.9, color='r', linestyle='-')

plotCumSumVariance(pca.explained_variance_ratio_)
 
plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 15})                    
plt.hist(df_new['T1'], bins = 9)
plt.xlabel("TIR")
plt.ylabel("Number of subject")
   
plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 15})                    
plt.hist(df_new['T2'], bins = 9)
plt.xlabel("TBR")
plt.ylabel("Number of subject")
        
plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 15})                    
plt.hist(df_new['T3'], bins = 9)
plt.xlabel("TAR")
plt.ylabel("Number of subject")

plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 15})                    
plt.hist(df_new['MG'], bins = 9)
plt.xlabel("MG")
plt.ylabel("Number of subject")

plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 15})                    
plt.hist(df_new['SDRC'], bins = 9)
plt.xlabel("SDRC")
plt.ylabel("Number of subject")

plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 15})                    
plt.hist(df_new['GMI'].drop(index = 17), bins = 9)
plt.xlabel("GMI")
plt.ylabel("Number of subject")

plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 15})                    
plt.hist(df_new['JIndex'].drop(index = 17), bins = 9)
plt.xlabel("J-index")
plt.ylabel("Number of subject")
             
plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 15})                    
plt.hist(df_new['after_eating'].drop(index = 17), bins = 9)
plt.xlabel("AE")
plt.ylabel("Number of subject")
             
        #np.append(TR(df_temp), np.array([np.mean(df_temp['glucose_mmol_L']), SDRC(df_temp), GMI(df_temp), Jindex(df_temp), after_eating(df_temp)]))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    ax.scatter(X_new[:,0], X_new[:,1], X_new[:,2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_new[:,0], X_new[:,1], X_new[:,2])
plt.show()
-------------------------------------------------------------------------
k = 1
eating = []
for j in range(1, df[df['copsacno'] == sub[k]].shape[0]):
    if (df['TimeSinceLastPicture'].loc[j] < df['TimeSinceLastPicture'].loc[j - 1]):
        eating.append(j)
 
filteres = lowess(df['Time'], df['glucose_mmol_L'], is_sorted=True, frac=0.015, it=0) #0.025
filtered = pd.to_datetime(filteres[:,0], format='%Y/%m/%dT%H:%M:%S')        
 
t = range(df[df['copsacno'] == sub[k]].shape[0])
g = df[df['copsacno'] == sub[k]]['glucose_mmol_L'].reset_index(drop = True)

plt.rcParams["figure.figsize"] = (15,3)
plt.plot(t, g, '.')
roots = eating

mark = [t.index(i) for i in roots]

plt.plot(roots,[g[i] for i in mark], ls="", marker="o", label="points")

plt.show()

----------------------------------------------------------------------------------
df['Time'] = pd.to_datetime(df['Time'], format = '%Y/%m/%dT%H:%M:%S')
df['Day'] = df['Time'].dt.date

k = 6
df1 = df[df['copsacno'] == sub[k]].reset_index(drop = True)
df1.shape
filteres = lowess(df1['glucose_mmol_L'], df1['Time'], is_sorted=True, frac=0.015, it=0) #0.025
filtered = pd.to_datetime(filteres[:,0], format='%Y/%m/%dT%H:%M:%S') 
# Set sizes

eating = []
eating_index = []
for j in range(1, df1.shape[0]):
    if (df1['TimeSinceLastPicture'].loc[j] < df1['TimeSinceLastPicture'].loc[j - 1]):
        eating_index.append(j)
        eating.append(df1['Time'].loc[j])           

glucose_mean = np.mean(df1['glucose_mmol_L'])
up = np.mean(df1['glucose_mmol_L']) + np.std(df1['glucose_mmol_L'])
dw = np.mean(df1['glucose_mmol_L']) - np.std(df1['glucose_mmol_L'])

plt.figure(figsize=(20,5))
plt.rcParams.update({'font.size': 20})

# Same plot as before
plt.plot(df1['Time'], df1['glucose_mmol_L'], '.')

# Plot 3 horizontal lines
plt.axhline(y=glucose_mean, color='pink', linestyle='-')
plt.axhline(y=up, color='pink', linestyle='-')
plt.axhline(y=dw, color='pink', linestyle='-')

mark = [range(df1['Time'].shape[0]).index(i) for i in eating_index]

plt.plot(eating,[df['glucose_mmol_L'][i] for i in mark], ls="", marker="o", label="points")

# Plot smoothed data
plt.plot(df1['Time'], filteres[:,1], 'r')

#Labels
plt.ylabel('Glucose')
plt.show()

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Z-score the features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# The PCA model
pca = PCA(n_components=2) # estimate only 2 PCs
X_new = pca.fit_transform(X) # project the original data into the PCA space
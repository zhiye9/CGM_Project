
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import datetime as datetime
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
import collections

#Read in CGM data
#df = pd.read_csv('../CGM_DATA/CGM_derX_withsleep.csv')
df = pd.read_csv('../CGM_DATA/CGM_withsleep.csv')
#df_sleep_withoutNAN = df.dropna(subset = 'is_sleep')
df = df.dropna(subset = 'is_sleep')
#df = df.rename(columns = {"Historic.Glucose..mmol.L.":"glucose_mmol_L",
#                            "cpno":"copsacno"})

#Change time format
df['Time'] = pd.to_datetime(df['Time'], format = '%Y/%m/%dT%H:%M:%S')
#Extract date info
df['Day'] = df['Time'].dt.date
df.reset

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

df_13d_index = [idx for idx, element in enumerate(glu_num) if (element > 1152) & (element < 1750)]

k = 5
df1 = df[df['copsacno'] == sub[k]].reset_index(drop = True)
df1 = df[df['copsacno'] == 321].reset_index(drop = True)

filteres = lowess(df1['glucose_mmol_L'], df1['Time'], is_sorted=True, frac=0.015, it=0) #0.025
filtered = pd.to_datetime(filteres[:,0], format='%Y/%m/%dT%H:%M:%S')           

glucose_mean = np.mean(df1['glucose_mmol_L'])
up = np.mean(df1['glucose_mmol_L']) + np.std(df1['glucose_mmol_L'])
dw = np.mean(df1['glucose_mmol_L']) - np.std(df1['glucose_mmol_L'])

plt.figure(figsize=(20,5))
plt.rcParams.update({'font.size': 20})

# Same plot as before
plt.plot(df1['Time'], df1['glucose_mmol_L'], '.')

plt.plot(df1['Time'], filteres[:,1], 'r')

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import datetime as datetime
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from itertools import compress

#Read in CGM data
df_raw = pd.read_csv('../CGM_DATA/CGM_derX_withsleep.csv')
df = pd.read_csv('../CGM_DATA/CGM_withsleep.csv')
#df_sleep_withoutNAN = df.dropna(subset = 'is_sleep')
df = df.dropna(subset = 'is_sleep')
#df = df.rename(columns = {"Historic.Glucose..mmol.L.":"glucose_mmol_L",
   #                         "cpno":"copsacno"})

#Change time format
df['Time'] = pd.to_datetime(df['Time'], format = '%Y/%m/%dT%H:%M:%S')
#Extract date info
df['Day'] = df['Time'].dt.date
df.reset_index(drop = True, inplace = True)

#Find the index of each subject
indexes = np.unique(df['copsacno'], return_index=True)[1]
sub = [df['copsacno'][index] for index in sorted(indexes)]

#Compute the number of glucose signal
glu_num = []
for i in range(len(sub)):
    glu_num.append(df[df['copsacno'] == sub[i]].shape[0]) 
glu_num_uni = np.unique(glu_num, return_counts=True)

#Plot distribution of #CGM
plt.figure(figsize=(15,5))
plt.ylabel("Number of subjects")
plt.xlabel("Number of glucose measurements")
arr = plt.hist(glu_num, bins = 10)
for i in range(10):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))

#Find subjects with more than 12 days and less than 18 days CGMs
df_12d_index = [idx for idx, element in enumerate(glu_num) if (element > 1152) & (element < 1750)]
df_12d = df[df['copsacno'].isin([sub[index] for index in df_12d_index])]

#Exclude subject with only inf in food image time
Non_all_inf = list(compress(np.unique(df_12d['copsacno']).tolist(), check_inf_bool))
df_12d_non_all_inf = df_12d[df_12d['copsacno'].isin(Non_all_inf)]

#Plot individual's CGM plot and eating time
k = 56
df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == Non_all_inf[k]].reset_index(drop = True)
#df1 = df_12d[df_12d['copsacno'] == 164].reset_index(drop = True)

filteres = lowess(df1['glucose_mmol_L'], df1['Time'], is_sorted=True, frac=0.015, it=0) #0.025
filtered = pd.to_datetime(filteres[:,0], format='%Y/%m/%dT%H:%M:%S')           

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

plt.plot(df1['Time'], df1['glucose_mmol_L'], '.')

mark = [range(df1['Time'].shape[0]).index(i) for i in eating_index]

plt.plot(eating,[df['glucose_mmol_L'][i] for i in mark], ls="", marker="o", label="points")

plt.plot(df1['Time'], filteres[:,1], 'r')

plt.ylabel('Glucose')
plt.show()

#Find #eating for each subject
eating_num = []
days = []
for i in range(len(Non_all_inf)):
    eating = []
    eating_index = []
    df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == Non_all_inf[i]].reset_index(drop = True)
    for j in range(1, df1.shape[0]):
        if (df1['TimeSinceLastPicture'].loc[j] < df1['TimeSinceLastPicture'].loc[j - 1]):
            eating_index.append(j)
            eating.append(df1['Time'].loc[j]) 
    eating_num.append(len(eating))
    days.append(len(np.unique(df1['Day'])))
    print("\r Process{0}%".format(round((i+1)*100/len(Non_all_inf))), end="")

#Check #eating versus days
len([j for j in [eating_num_i - days_i*2 for eating_num_i, days_i in zip(eating_num, days)] if j > 0])
plt.figure(figsize=(10,10))
plt.ylabel('Eating number')
plt.xlabel('Days')
x = np.linspace(13, 18, 5)
plt.plot(x, x , '-r')
plt.scatter(days, eating_num) 
plt.xlim(xmin=13)
plt.ylim(ymin=0)
plt.show()






----------------------------------------------------------------------------------------------------------------------------------------------
check_inf = [np.unique(~np.isfinite(df_12d[df_12d['copsacno'] == i]['TimeSinceLastPicture']), return_counts = True) for i in np.unique(df_12d['copsacno'])]
check_inf_bool = [~(~np.isfinite(df_12d[df_12d['copsacno'] == i]['TimeSinceLastPicture'])).all() for i in np.unique(df_12d['copsacno'])]
np.unique(df_12d['copsacno']).tolist()[check_inf_bool]

len(list(compress(np.unique(df_12d['copsacno']).tolist(), check_inf_bool)))

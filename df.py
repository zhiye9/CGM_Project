import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import datetime as datetime
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from itertools import compress
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib
import scipy

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score 
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr as pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import warnings
import time

import sys

sys.path.append('/home/zhi/data/CGM/VBA/VBA-OMM/Python/')
import VBA_OMM

df_diet = pd.read_csv('~/data/CGM/CGM_DATA/NAS_data/Diet_chanllenge_BiochemData.csv')
df_diet = df_diet.rename(columns = {"COPSACNO":"copsacno"})
df_diet = df_diet.tail(df_diet.shape[0] - 4)
df_diet['copsacno'] = df_diet['copsacno'].astype(str)
df_diet.reset_index(drop = True, inplace = True)
df_diet = df_diet[df_diet.columns[3:]]
df_diet = df_diet.drop('COMMENTS', axis = 1)

df_diet = df_diet[['copsacno', 'TIMEPOINT', 'Glucose']]
df_diet['Glucose'] = df_diet['Glucose'].astype(float)
df_diet = df_diet.sort_values(['copsacno', 'TIMEPOINT'])

df_diet2 = pd.read_csv('~/data/CGM/CGM_DATA/NAS_data/bloodsample.csv')
df_diet2 = df_diet2[df_diet2["Insulin"].str.contains("Hæmolyse") == False]
df_diet2 = df_diet2[df_diet2["Insulin"].str.contains("NS") == False]
df_diet2 = df_diet2[df_diet2["Insulin"].str.contains("Mislykket") == False]
df_diet2['Insulin'] = df_diet2['Insulin'].astype(float)

df_diet2 = df_diet2[df_diet2["C-peptid"].str.contains('<16,6') == False]
#df_diet2.loc[df_diet2['C-peptid'] == '<16,6', 'C-peptid'] = 16.6
df_diet2['C-peptid'] = df_diet2['C-peptid'].astype(float)
df_diet2['C-peptid_nmol/L'] = df_diet2['C-peptid']/1000
df_diet2[['copsacno','TIMEPOINT']] = df_diet2['ID'].str.split('_',expand=True)
df_diet2['age'] = df_diet2['age'].astype(float)
df_diet2 = df_diet2[['copsacno', 'TIMEPOINT', 'age', 'Insulin', 'C-peptid_nmol/L']]
df_diet2['copsacno'] = df_diet2['copsacno'].astype(str)
df_diet2['TIMEPOINT'] = df_diet2['TIMEPOINT'].astype(float)

df_glu_insulin = pd.merge(df_diet2, df_diet, on = ['copsacno', 'TIMEPOINT'])
df_glu_insulin = df_glu_insulin.rename(columns={'C-peptid_nmol/L': 'C-peptid'})

df_glu_insulin = df_glu_insulin.sort_values(['copsacno', 'TIMEPOINT'])
df_glu_insulin_8 = df_glu_insulin.groupby('copsacno').filter(lambda x: len(x) == 8)
#df_glu_insulin_8.to_csv('/home/zhi/data/CGM/CGM_DATA/df_glu_insulin_Cpeptid.csv', index = False)
#df_glu_insulin_8.to_csv('/home/zhi/data/CGM/CGM_DATA/df_biochem_insulin_Cpeptid.csv', index = False)
df_glu_insulin_8 = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df_biochem_insulin_Cpeptid.csv')

df_glu_insulin_Cpeptid = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df_glu_insulin_Cpeptid.csv')
df_glu_insulin_Cpeptid['copsacno'] = df_glu_insulin_Cpeptid['copsacno'].astype(str)

#drop all columns with % in column names
df_biochem_glu_insulin_8 = df_glu_insulin_8.loc[:,~df_glu_insulin_8.columns.str.contains('%')]
df_biochem_glu_insulin_8.drop(columns = ['SAMPLEDATE'], inplace = True)
df_biochem_glu_insulin_8 = df_biochem_glu_insulin_8.loc[:,~df_biochem_glu_insulin_8.columns.str.contains('Clinical')]
df_biochem_glu_insulin_8 = df_biochem_glu_insulin_8.loc[:,~df_biochem_glu_insulin_8.columns.str.contains('/')]
df_biochem_glu_insulin_8.drop(columns = ['Phe'], inplace = True)
df_biochem_glu_insulin_8 = df_biochem_glu_insulin_8.loc[:,~df_biochem_glu_insulin_8.columns.str.contains('size')]
df_biochem_glu_insulin_8.drop(columns = ['Glycerol'], inplace = True)
df_biochem_glu_insulin_8.to_csv('/home/zhi/data/CGM/CGM_DATA/df_biochem_insulin_Cpeptid_filtered.csv', index = False)

test1 = df_glu_insulin[df_glu_insulin['copsacno'] == '95'].sort_values('TIMEPOINT')[['copsacno', 'TIMEPOINT', 'Glucose', 'Insulin']]
test2 = df_glu_insulin[df_glu_insulin['copsacno'] == '101'].sort_values('TIMEPOINT')[['copsacno', 'TIMEPOINT', 'Glucose', 'Insulin']]

df_image_time = pd.read_csv('../CGM_DATA/images-v1-copsac-approxtimetaken.xlsx')
'''
#Read in CGM data
df_CGM =pd.read_csv('../CGM_DATA/CGM01202022.csv')
df_CGM = df_CGM.rename(columns = {"Historic.Glucose..mmol.L.":"glucose_mmol_L", "cpno":"copsacno"})
df_CGM_cpno = df_CGM.dropna(subset = 'copsacno')
df_CGM_cpno['copsacno'] = df_CGM_cpno['copsacno'].astype(int).astype(str)
df_CGM_cpno.reset_index(drop = True, inplace = True)

df = pd.read_csv('../CGM_DATA/CGM_withsleep.csv')
df['copsacno'] = df['copsacno'].astype(str)
df.reset_index(drop = True, inplace = True)

df_CGM_food_sleep = df.dropna(subset = 'is_sleep')
df_CGM_food_sleep.reset_index(drop = True, inplace = True)

df_pheno = pd.read_csv('../CGM_DATA/CGM3.csv')

df_sleep = pd.read_csv('../CGM_DATA/CGM_derX_withsleep.csv')
df_sleep = df_sleep.rename(columns = {"cpno":"copsacno"})
df_sleep_noNAN = df_sleep.dropna(subset = 'is_sleep')
df_sleep_noNAN['copsacno'] = df_sleep_noNAN['copsacno'].astype(int).astype(str)
df_sleep_noNAN.reset_index(drop = True, inplace = True)

df_diet = pd.read_csv('../CGM_DATA/NAS_data/Diet_chanllenge_BiochemData.csv')
df_diet = df_diet.rename(columns = {"COPSACNO":"copsacno"})
df_diet_noheader = df_diet.tail(df_diet.shape[0] - 4)
df_diet_noheader['copsacno'] = df_diet_noheader['copsacno'].astype(str)
df_diet_noheader.reset_index(drop = True, inplace = True)

df_CGM_diet = pd.merge(df_CGM_cpno, df_diet_noheader, on = 'copsacno')
df_CGM_diet.reset_index(drop = True, inplace = True)

df_CGM_diet_food = pd.merge(df, df_diet_noheader, on = 'copsacno')
df_CGM_diet_food.reset_index(drop = True, inplace = True)

df_diet_sleep = pd.merge(df_sleep_noNAN, df_diet_noheader, on = 'copsacno')

df_sleep = df.dropna(subset = 'is_sleep')
df_CGM_diet_food_sleep = pd.merge(df_sleep, df_diet_noheader, on = 'copsacno')


df_CP = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/NAS_data/bloodsample.csv')

'''
df = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/CGM_withsleep.csv')
df['copsacno'] = df['copsacno'].astype(str)
df.reset_index(drop = True, inplace = True)

df_CGM =pd.read_csv('/home/zhi/data/CGM/CGM_DATA/CGM01202022.csv')
df_CGM = df_CGM.rename(columns = {"Historic.Glucose..mmol.L.":"glucose_mmol_L", "cpno":"copsacno"})
df_CGM_cpno = df_CGM.dropna(subset = 'copsacno')
df_CGM_cpno['copsacno'] = df_CGM_cpno['copsacno'].astype(int).astype(str)
df_CGM_cpno.reset_index(drop = True, inplace = True)

acc_list = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/accelerometer_data.txt', header = None, names = ['file_name'])
acc_list['copsacno'] = [int(acc_list['file_name'].loc[i][2:-19]) for i in range(acc_list.shape[0])]
acc_list['copsacno'] = acc_list['copsacno'].astype(str)
u, c = np.unique(acc_list['copsacno'], return_counts = True)
dup = u[c > 1]
acc_list_norep = acc_list.drop_duplicates(subset=['copsacno'])

df_diet = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/NAS_data/Diet_chanllenge_BiochemData.csv')
df_diet = df_diet.rename(columns = {"COPSACNO":"copsacno"})
df_diet_noheader = df_diet.tail(df_diet.shape[0] - 4)
df_diet_noheader['copsacno'] = df_diet_noheader['copsacno'].astype(str)
df_diet_noheader.reset_index(drop = True, inplace = True)
df_CGM_diet = pd.merge(df_CGM_cpno, df_diet_noheader, on = 'copsacno')
df_CGM_diet.reset_index(drop = True, inplace = True)

#df_CGM_acc = pd.merge(df_CGM_cpno[['copsacno', 'glucose_mmol_L']], acc_list_norep, on = 'copsacno')
CGM_acc = np.intersect1d(np.unique(df_CGM_cpno['copsacno']), np.unique(acc_list_norep['copsacno']))
CGM_acc_food = np.intersect1d(CGM_acc, np.unique(df['copsacno']))
CGM_diet_acc = np.intersect1d(np.unique(acc_list_norep['copsacno']), np.unique(df_CGM_diet['copsacno']))
CGM_acc_food_diet = np.intersect1d(CGM_diet_acc, CGM_acc_food)

df_CGM_acc_food_diet = df[df.copsacno.isin(CGM_acc_food_diet)]

#df_CGM_cpno['Time'] = pd.to_datetime(df_CGM_cpno['Time'], format = '%Y/%m/%dT%H:%M:%S')

df_pheno = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/CGM3.csv')
df_pheno['copsacno'] = df_pheno['copsacno'].astype(str)
df_pheno_gorup = df_pheno.drop_duplicates(subset = 'copsacno')
df_pheno_gorup_control = df_pheno_gorup.dropna(subset = ['Sex_binary', 'Race', 'householdincome', 'm_edu', 'Mother_age_2yrs'])
df_control = df_pheno_gorup.dropna(subset = ['Sex_binary', 'Race', 'householdincome', 'm_edu', 'Mother_age_2yrs'])[['copsacno', 'Sex_binary', 'Race', 'householdincome', 'm_edu', 'Mother_age_2yrs']]
df_BMI = df_pheno_gorup_control.dropna(subset = 'bmi18y')[['copsacno', 'bmi18y']]
df_waisthipratio = df_pheno_gorup_control.dropna(subset = 'waisthipratio')[['copsacno', 'waisthipratio']]
df_FITNESS = df_pheno_gorup_control.dropna(subset = 'FITNESS')[['copsacno', 'FITNESS']]
df_adhd_score = df_pheno_gorup_control.dropna(subset = 'adhd_score')[['copsacno', 'adhd_score']]
df_Musclefatratio = df_pheno_gorup_control.dropna(subset = 'Musclefatratio')[['copsacno', 'Musclefatratio']]
df_Musclemass = df_pheno_gorup_control.dropna(subset = 'Muscle_mass')[['copsacno', 'Muscle_mass']]

df = df_CGM_acc_food_diet
df = df_CGM_cpno
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

glu_num_days = [i/4/24 for i in glu_num]

#Plot distribution of #CGM
plt.figure(figsize=(12,5), dpi = 1000)
plt.ylabel("Number of subjects")
#plt.xlabel("Number of glucose measurements")
plt.xlabel("Number of CGM days")
arr = plt.hist(glu_num_days, bins = 13)
#arr = plt.hist(glu_num, bins = 10)
for i in range(10):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))

#Find subjects with more than 12 days and less than 18 days CGMs
df_12d_index = [idx for idx, element in enumerate(glu_num) if (element > 1152) & (element < 1750)]
df_12d = df[df['copsacno'].isin([sub[index] for index in df_12d_index])]
df_12d.reset_index(drop = True, inplace = True)

#Exclude subject with only inf in food image time
check_inf_bool = [~(~np.isfinite(df_12d[df_12d['copsacno'] == i]['TimeSinceLastPicture'])).all() for i in np.unique(df_12d['copsacno'])]
Non_all_inf = list(compress(np.unique(df_12d['copsacno']).tolist(), check_inf_bool))
df_12d_non_all_inf = df_12d[df_12d['copsacno'].isin(Non_all_inf)]
df_12d_non_all_inf['copsacno_Day'] = df_12d_non_all_inf[['copsacno', 'Day']].astype(str).agg('-'.join, axis=1)

----------------------------------------------------------------------------------------------------------------------
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

df_new = pd.DataFrame(0, index = np.arange(len(df_12d_index)), columns = ['copsacno','T1', 'T2', 'T3', 'SDRC', 'GMI', 'JIndex'])
df_new['copsacno'] = [sub[index] for index in df_12d_index]

for q in range(len(df_12d_index)):
    print(q)
#for q in range(10):
    T1 = []
    T2 = []
    T3 = []
    MG = []
    Src = []
    gmii = []
    ji = []
    for t in range(0, 1151, 96):
        df_temp = df_CGM_cpno[df_CGM_cpno['copsacno'] == sub[df_12d_index[q]]].reset_index(drop = True).loc[t:t+95].reset_index(drop = True)
        T1.append(TR(df_temp)[0])
        T2.append(TR(df_temp)[1])
        T3.append(TR(df_temp)[2])
        MG.append(np.mean(df_temp['glucose_mmol_L']))
        Src.append(SDRC(df_temp))
        gmii.append(GMI(df_temp))
        ji.append(Jindex(df_temp)) 
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
    df_new['SDRC'].loc[q] = np.mean(Src)
    df_new['GMI'].loc[q] = np.mean(gmii)
    df_new['JIndex'].loc[q] = np.mean(ji)
    print("\r Process{}%".format(round((q+1)*100/len(df_12d_index))), end="")

df_new_outcome = pd.merge(df_BMI, df_new, on = 'copsacno')
df_new_outcome = pd.merge(df_Musclefatratio, df_new, on = 'copsacno')
df_new_outcome = pd.merge(df_FITNESS, df_new, on = 'copsacno')
df_new_outcome = pd.merge(df_waisthipratio, df_new, on = 'copsacno')
df_new_outcome = pd.merge(df_adhd_score, df_new, on = 'copsacno')
df_new_outcome = pd.merge(df_Musclemass, df_new, on = 'copsacno')

df_new_outcome = pd.merge(df_new_outcome, df_control, on = 'copsacno')
df_new_outcome = pd.get_dummies(df_new_outcome, columns=['Race'], drop_first = True)

scaler = StandardScaler()
#x_var = ['T1', 'T2', 'T3', 'SDRC', 'GMI', 'JIndex']
x_var = ['T1', 'T2', 'T3', 'SDRC', 'GMI', 'JIndex', 'householdincome', 'm_edu', 'Mother_age_2yrs']
scaler.fit(np.array(df_new_outcome[x_var]))
X = scaler.transform(np.array(df_new_outcome[x_var])) 

def CV(p_grid, out_fold, in_fold, model, X, y, rand, n_beta = False):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        if (n_beta):
            beta.append(clf.best_estimator_.coef_[:n_beta])
        #print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train)
        #print(y_pred)
        r2train.append(r2_score(y_train, y_pred))
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        #print(y_pred)
        r2test.append(r2_score(y_test, y_pred))
        
    if (n_beta):
        return r2train, r2test, beta
    else:
        return r2train, r2test

#Set parameters of cross-validation
par_grid = {'alpha': [1e-2, 3e-2, 5e-2, 1e-1, 1, 1e-3, 0.3]}
par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
rand = 9

#X_reg = np.concatenate((X, np.array(df_new_outcome[['Sex_binary']])), axis = 1)
X_reg = np.concatenate((X, np.array(df_new_outcome[['Sex_binary']]), np.array(df_new_outcome[['Race_non-caucasian']])), axis = 1)
BMI_train, BMI_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_new_outcome['bmi18y']), n_beta = False, rand = 66)
BMI_train, BMI_test, beta_BMI = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['bmi18y']), n_beta = 11, rand = 76)
MFratio_train, MFratio_test, beta_MFratio = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['Musclefatratio']), n_beta = 8, rand = 99)
FatPer_train, FatPer_test, beta_FatPer = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['Fat_percent']), n_beta = 11, rand = 99)
MuscleMass_train, MuscleMass_test, beta_MuscleMass = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['Muscle_mass']), n_beta = 11, rand = 99)
MuscleMass_train, MuscleMass_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_new_outcome['Muscle_mass']), n_beta = False, rand = 66)

Fit_train, Fit_test, beta_Fit = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['FITNESS']), n_beta = 11, rand = 66)
Fit_train, Fit_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_new_outcome['FITNESS']), n_beta = False, rand = 9)
WHR_train, WHR_test, beta_Fit = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['waisthipratio']), n_beta = 8, rand = 66)
ADHD_train, ADHD_test, beta_Fit = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['adhd_score']), n_beta = 8, rand = 66)


pd.DataFrame(beta_MFratio).T.to_csv('MFratio.csv')
pd.DataFrame(beta_FatPer).T.to_csv('FatPer.csv')
pd.DataFrame(beta_MuscleMass).T.to_csv('MuscleMass.csv')
pd.DataFrame(beta_Fit).T
model.fit(x_train, y_train)
model.predict(x_train)

labels = ['Muscle_fat_ratio', 'Bone_mass','Muscle_mass']
dpi = 600
title = 'Regression results of outcomes and CGM features'

x = np.arange(len(labels))
train_mean = [np.mean(MFratio_train), np.mean(Bonemass_train), np.mean(MuscleMass_train)]
test_mean = [np.mean(MFratio_test), np.mean(Bonemass_test), np.mean(MuscleMass_test)]
train_std = [np.std(MFratio_train),  np.std(Bonemass_train), np.std(MuscleMass_train)]
test_std = [np.std(MFratio_test), np.std(Bonemass_test), np.std(MuscleMass_test)]    

#train_mean = [np.mean(Sk_train), np.mean(MFratio_train), np.mean(Bonemass_train), np.mean(MuscleMass_train)]
#test_mean = [np.mean(Sk_test), np.mean(MFratio_test), np.mean(Bonemass_test), np.mean(MuscleMass_test)]
#train_std = [np.std(Sk_train), np.std(MFratio_train),  np.std(Bonemass_train), np.std(MuscleMass_train)]
#test_std = [np.std(Sk_test), np.std(MFratio_test), np.std(Bonemass_test), np.std(MuscleMass_test)]    

fig, ax = plt.subplots(dpi = 1000)

width = 0.35
rects1 = ax.bar(x - width/2, train_mean, width, yerr = train_std, label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, test_mean, width, yerr = test_std, label='test', align='center', ecolor='black', capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
#plt.xticks(rotation=60)
#plt.yticks(np.linspace(0.0,0.7,0.1))
ax.set_xticklabels(labels, fontsize = 10)
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
#for i, v in enumerate(train_mean):
   # rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
#for i, v in enumerate(test_mean):
    #rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
fig.tight_layout()

plt.show()   
----------------------------------------------------------------------------------------------------------------------

#Plot individual's CGM plot and eating time
def plot_indi(k, eatdot = True, resolu = 1000):
    df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == Non_all_inf[k]].reset_index(drop = True)
    #df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == k].reset_index(drop = True)
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

    plt.figure(figsize=(20,5), dpi = resolu)
    plt.rcParams.update({'font.size': 20})

    colors = ['purple','dodgerblue']
    plt.scatter(df1['Time'], df1['glucose_mmol_L'], marker = '.', c = pd.get_dummies(df1['is_sleep'])['awake'].astype(int).to_list(), cmap = matplotlib.colors.ListedColormap(colors))
    
    mark = [range(df1['Time'].shape[0]).index(i) for i in eating_index]

    if (eatdot):
        plt.plot(eating,[df1['glucose_mmol_L'][i] for i in mark], ls="", marker="o", label="points", c = 'orange')
    else:
        for xv in eating:
            plt.axvline(x = xv, color = 'orange')
    
    plt.plot(df1['Time'], filteres[:,1], 'r')
    plt.ylabel('Glucose (mmol/L)')
    
    red_line = Line2D([0], [0], label='Lowess smoothing line', color='r')
    blue_patch = mpatches.Patch(color='blue', label='Awake')
    purple_patch = mpatches.Patch(color='purple', label='Sleep')
    orange_patch = mpatches.Patch(color='Orange', label='Food image')
    orange_line = Line2D([0], [0], label='Eating (Taking photo)', color='orange')
    if (eatdot):
        #plt.legend(handles=[red_line, blue_patch, orange_patch], bbox_to_anchor=(1, 1), prop={'size': 15})
        plt.legend(handles=[red_line, blue_patch, purple_patch, orange_patch], bbox_to_anchor=(1, 1.2), prop={'size': 15})
    else:
        plt.legend(handles=[red_line, blue_patch, orange_line], bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.show()

def plot_indi_CGM(k, eatdot = True, resolu = 1000):
    df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == Non_all_inf[k]].reset_index(drop = True)
    #df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == k].reset_index(drop = True)
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

    plt.figure(figsize=(20,5), dpi = resolu)
    plt.rcParams.update({'font.size': 20})

    colors = ['purple','dodgerblue']
    #plt.scatter(df1['Time'], df1['glucose_mmol_L'], marker = '.', c = pd.get_dummies(df1['is_sleep'])['awake'].astype(int).to_list(), cmap = matplotlib.colors.ListedColormap(colors))
    plt.scatter(df1['Time'], df1['glucose_mmol_L'], marker = '.'
    mark = [range(df1['Time'].shape[0]).index(i) for i in eating_index]

    if (eatdot):
        plt.plot(eating,[df1['glucose_mmol_L'][i] for i in mark], ls="", marker="o", label="points", c = 'orange')
    else:
        for xv in eating:
            plt.axvline(x = xv, color = 'orange')
    
    plt.plot(df1['Time'], filteres[:,1], 'r')
    plt.ylabel('Glucose (mmol/L)')
    
    red_line = Line2D([0], [0], label='Lowess smoothing line', color='r')
    blue_patch = mpatches.Patch(color='blue', label='Awake')
    #purple_patch = mpatches.Patch(color='purple', label='Sleep')
    orange_patch = mpatches.Patch(color='Orange', label='Food image')
    orange_line = Line2D([0], [0], label='Eating (Taking photo)', color='orange')
    if (eatdot):
        #plt.legend(handles=[red_line, blue_patch, orange_patch], bbox_to_anchor=(1, 1), prop={'size': 15})
        plt.legend(handles=[red_line, blue_patch, purple_patch, orange_patch], bbox_to_anchor=(1, 1.2), prop={'size': 15})
    else:
        plt.legend(handles=[red_line, blue_patch, orange_line], bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.show()


k = Non_all_inf.index('303')
Non_all_inf[k]
plot_indi(k, eatdot = True, resolu = 860)

for k in range(60, 70):
    plot_indi(67, eatdot = True)

tete = pd.read_csv('/nas/data/users/david/All_CSVs/CP376_2019-02-18_RAW.csv', skiprows = 10)
tete = pd.read_csv('/nas/data/users/david/All_CSVs/CP72_2017-06-20_RAW.csv', skiprows = 10)
tete = pd.read_csv('/nas/data/users/david/All_CSVs/CP305_2018-05-14_RAW.csv', skiprows = 10)
tete = pd.read_csv('/nas/data/users/david/All_CSVs/CP303_2018-06-27_RAW.csv', skiprows = 10)
plt.plot(tete['Accelerometer X'])
plt.plot(tete['Accelerometer Y'])
plt.plot(tete['Accelerometer Z'])

#Find #eating for each subject with more than 12d
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

plt.figure(figsize=(9,9))
plt.ylabel('Eating number')
plt.xlabel('Days')
x = np.linspace(13, 19, 5)
plt.plot(x, x , '-r')
plt.scatter(days, eating_num) 
plt.xlim(xmin=13)
plt.ylim(ymin=0)
plt.show()

#Check #eating versus days
len([j for j in [eating_num_i // days_i for eating_num_i, days_i in zip(eating_num, days)] if j >= 1])
len([j for j in [eating_num_i // days_i for eating_num_i, days_i in zip(eating_num, days)] if j >= 2])
len([j for j in [eating_num_i // days_i for eating_num_i, days_i in zip(eating_num, days)] if j >= 3])

#Find #eating for each subject per day with more than 12d
Non_all_inf_cpno_Day = np.unique(df_12d_non_all_inf['copsacno_Day'])

eating_num = []
for i in range(len(Non_all_inf_cpno_Day)):
    eating = []
    eating_index = []
    df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno_Day'] == Non_all_inf_cpno_Day[i]].reset_index(drop = True)
    for j in range(1, df1.shape[0]):
        if (df1['TimeSinceLastPicture'].loc[j] < df1['TimeSinceLastPicture'].loc[j - 1]):
            eating_index.append(j)
            eating.append(df1['Time'].loc[j]) 
    eating_num.append(len(eating))
    print("\r Process{0}%".format(round((i+1)*100/len(Non_all_inf_cpno_Day))), end="")

#Check #eating versus days
len([i for i in eating_num if i >= 2])

#Plot #eating
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(eating_num, bins = 12)
ax.set_xticks(bins)
ax.set_xlabel('Eating times per day', fontsize=15)
ax.set_ylabel('Number of CGM days', fontsize=15)
for count, patch in zip(counts,patches):
    ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()), fontsize = 12)
plt.show()

#Combine cpnp and day info
df['copsacno_Day'] = df[['copsacno', 'Day']].astype(str).agg('-'.join, axis=1)
#Exclude subject with only inf in food image time
check_inf_bool_all = [~(~np.isfinite(df[df['copsacno'] == i]['TimeSinceLastPicture'])).all() for i in np.unique(df['copsacno'])]
Non_all_inf_all = list(compress(np.unique(df['copsacno']).tolist(), check_inf_bool_all))
df_non_all_inf = df[df['copsacno'].isin(Non_all_inf_all)]
Non_all_inf_all_cpno_Day = np.unique(df_non_all_inf['copsacno_Day'])

#F#eating for each subject per day with all 143 subjects
eating_num_all = []
for i in range(len(Non_all_inf_all_cpno_Day)):
    eating = []
    eating_index = []
    df1 = df_non_all_inf[df_non_all_inf['copsacno_Day'] == Non_all_inf_all_cpno_Day[i]].reset_index(drop = True)
    for j in range(1, df1.shape[0]):
        if (df1['TimeSinceLastPicture'].loc[j] < df1['TimeSinceLastPicture'].loc[j - 1]):
            eating_index.append(j)
            eating.append(df1['Time'].loc[j]) 
    
    eating_num_all.append(len(eating))
    print("\r Process{0}%".format(round((i+1)*100/len(Non_all_inf_all_cpno_Day))), end="")


len([i for i in eating_num_all if i >= 2])

#Plot #eating
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(eating_num_all, bins = 12)
ax.set_xticks(bins)
ax.set_xlabel('Eating times per day', fontsize=15)
ax.set_ylabel('Number of CGM days', fontsize=15)
for count, patch in zip(counts,patches):
    ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()), fontsize = 12)
plt.show()

#Segmentation
df_non_all_inf['eating'] = ['not_eat']*df_non_all_inf.shape[0]
df_non_all_inf = df_non_all_inf.reset_index(drop = True)
for j in range(1, df_non_all_inf.shape[0]):
    if (df_non_all_inf['TimeSinceLastPicture'].loc[j] < df_non_all_inf['TimeSinceLastPicture'].loc[j - 1]) and (df_non_all_inf['copsacno'].loc[j] == df_non_all_inf['copsacno'].loc[j - 1]):
        df_non_all_inf['eating'].loc[j] = 'eat'

df_2meals_index = [idx for idx, element in enumerate(eating_num_all) if (element >= 2)]
df_2meals = df_non_all_inf[df_non_all_inf['copsacno_Day'].isin([Non_all_inf_all_cpno_Day[index] for index in df_2meals_index])]
df_2meals = df_2meals.reset_index(drop = True)

df_2meals[['last_1','next_1', 'next_2', 'next_3', 'next_4']] = np.nan
for i in range(len(df_2meals_index)):
    df_2meals_cpno_day_index = df_2meals.index[df_2meals['copsacno_Day'] == Non_all_inf_all_cpno_Day[df_2meals_index[i]]].tolist()
    for j in range(df_2meals_cpno_day_index[1], df_2meals_cpno_day_index[-1] - 3):
        df_2meals['last_1'].loc[j] = df_2meals['glucose_mmol_L'].loc[j - 1]
        df_2meals['next_1'].loc[j] = df_2meals['glucose_mmol_L'].loc[j + 1]
        df_2meals['next_2'].loc[j] = df_2meals['glucose_mmol_L'].loc[j + 2]
        df_2meals['next_3'].loc[j] = df_2meals['glucose_mmol_L'].loc[j + 3]
        df_2meals['next_4'].loc[j] = df_2meals['glucose_mmol_L'].loc[j + 4]
    
    print("\r Process{0}%".format(round((i+1)*100/len(df_2meals_index))), end="")

df_X = df_2meals.dropna(how = 'any', subset = df_2meals.columns[-5:])

#df_X.to_csv('df_X_6glu.csv', index = False)
df_X = pd.read_csv('../CGM_DATA/df_X_6glu.csv')

X = np.array(df_X[['last_1', 'glucose_mmol_L', 'next_1', 'next_2', 'next_3', 'next_4']])
y = np.asarray(pd.get_dummies(df_X['eating'], dtype = int)['eat'])

accuracy = []
f1 = []
recall = []
precision_plot = []
recall_plot = []
pr_thr = []
precision = []
spcificty = []
tprs = []
fprs = []
roc_thr = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
AUROC = []
AUPR = []
y_true = []
y_proba = []

p_grid = {'svc__C': [0.1, 1, 5, 10, 50, 100]}
model = SVC(kernel = 'linear',  probability = True, class_weight = 'balanced')

p_grid = {'randomforestclassifier__max_depth': [10, 50, 100, 200, None]}
model = RandomForestClassifier()

p_grid = {'KNeighborsClassifier__n_neighbors': [3, 5, 7, 10, 12]}
model = KNeighborsClassifier(weights = 'distance')

outer_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)
inner_cv = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)

for j in range(5):
    train, test = list(outer_cv.split(X, y))[j]
    x_train, x_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    # fit the classifier
    classifier = make_pipeline(StandardScaler(), model)
    clf = GridSearchCV(estimator = classifier, param_grid = p_grid, cv = inner_cv, scoring = "balanced_accuracy")
    clf.fit(x_train, y_train)
    print(clf.best_estimator_)
    # make predictions for the left-out test subjects
    y_pred = clf.predict(x_test)
    print(j)
    a = metrics.accuracy_score(y_test, y_pred)
    f = metrics.f1_score(y_test, y_pred, pos_label = 1)
    r = metrics.recall_score(y_test, y_pred, average='binary', pos_label = 1)
    p = metrics.precision_score(y_test, y_pred, average='binary', pos_label = 1)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    s = tn / (tn+fp)
    #print(a)
    #print(f)
    accuracy.append(a)
    f1.append(f)
    recall.append(r)
    precision.append(p)
    spcificty.append(s)
    y_score = clf.predict_proba(x_test)[:,1]
    y_true.append(y_test) 
    y_proba.append(y_score)
    pre, re, thresholds = precision_recall_curve(y_test, y_score, pos_label = 1)
    precision_plot.append(pre)
    recall_plot.append(re)
    pr_thr.append(thresholds)
    AUPR.append(auc(re, pre))
    fpr, tpr, thresholds = roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)
    fprs.append(fpr)
    tprs.append(tpr)
    roc_thr.append(thresholds)
    AUROC.append(auc(fpr, tpr))
print(np.mean(accuracy))
print(np.std(accuracy))

print(np.mean(f1))
print(np.std(f1))

print(np.mean(recall))
print(np.std(recall))

print(np.mean(precision))
print(np.std(precision))

print(np.mean(spcificty))
print(np.std(spcificty))

print(np.mean(AUROC))
print(np.std(AUROC))

print(np.mean(AUPR))
print(np.std(AUPR))

plt.figure()
lw = 2
for i in range(5):
    plt.plot(
        fprs[i],
        tprs[i],
        lw=lw,
        label="Fold %s ROC curve (area = %0.2f)" % (i, AUROC[i]),
    )
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
fpr_all, tpr_all, thresholds_all = roc_curve(np.concatenate(y_true, axis=0), np.concatenate(y_proba, axis=0), drop_intermediate = False, pos_label = 1)
plt.plot(fpr_all, tpr_all, lw=lw, label="Overall ROC curve (area = %0.2f)" % auc(fpr_all, tpr_all))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC for KNN")
plt.legend(loc="lower right")
plt.show()

plt.figure()
lw = 2
for i in range(5):
    plt.plot(
        recall_plot[i],
        precision_plot[i],
        lw=lw,
        label="Fold %s PR curve (area = %0.2f)" % (i, AUPR[i]),
    )
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
pre_all, re_all, thresholds_pr_all = precision_recall_curve(np.concatenate(y_true, axis=0), np.concatenate(y_proba, axis=0), pos_label = 1)
plt.plot(re_all, pre_all, lw=lw, label="Overall PR curve (area = %0.2f)" % auc(re_all, pre_all))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR curve for KNN")
plt.legend(loc="lower right")
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------
check_inf = [np.unique(~np.isfinite(df_12d[df_12d['copsacno'] == i]['TimeSinceLastPicture']), return_counts = True) for i in np.unique(df_12d['copsacno'])]
np.unique(df_12d['copsacno']).tolist()[check_inf_bool]

len(list(compress(np.unique(df_12d['copsacno']).tolist(), check_inf_bool)))
----------------------------------------------------------------------------------------------------------------------------------------------

#import neurokit2 as nk
time_glu = pd.read_excel('images-v1-dl-approxtimetake-glucosecgm_v2.xlsx', sheet_name='timephone')
time_glu['Folder'] = time_glu['Folder'].astype(str)

outcome = pd.read_excel('images-v1-dl-outcomes.xlsx')  
outcome.rename(columns = {'Unnamed: 0':'Folder'}, inplace = True)
outcome['Folder'] = outcome['Folder'].astype(str)

time_glu_nonan = time_glu.dropna(subset = time_glu.columns[-13:])
time_glu_nonan = time_glu_nonan.reset_index(drop = True)

indexes = np.unique(time_glu_nonan['Folder'], return_index=True)[1]
sub = [time_glu_nonan['Folder'][index] for index in sorted(indexes)]

glu_num = []
for i in range(len(sub)):
    glu_num.append(time_glu_nonan[time_glu_nonan['Folder'] == sub[i]].shape[0])

#max_index = glu_num.index(max(glu_num))

#temp = time_glu_nonan[time_glu_nonan['Folder'] == sub[69]]

glu_num_uni = np.unique(glu_num, return_counts=True)

plt.figure(figsize=(15,5))
plt.ylabel("Number of subjects")
plt.xlabel("Number of glucose measurements")
arr = plt.hist(glu_num, bins = 10)
for i in range(10):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))

df_3d_index = [idx for idx, element in enumerate(glu_num) if (element > 3)]

def TR(x, sd = 1, sr=15):
    up = np.mean(x[-13:]) + sd*np.std(x[-13:])
    dw = np.mean(x[-13:]) - sd*np.std(x[-13:])
    TIR = len(x[-13:][(x[-13:]<= up) & (x[-13:]>= dw)])/13
    TBR = len(x[-13:][x[-13:] < dw])/13
    TAR = len(x[-13:][x[-13:] > up])/13
    return TIR, TBR, TAR

def SDRC(x):
    SDRC = np.abs(x[-13:].diff()/15)
    return np.std(SDRC)

def GMI(x):
    GMI = 12.71 + (4.70587*np.mean(x[-13:]))
    return GMI    

def Jindex(x):
    J = 0.324*((np.mean(x[-13:])+np.std(x[-13:]))**2)
    return J

def slope_after_eating(x):
    return abs(x[-1] - x[-5])/60

df_new = pd.DataFrame(0, index = np.arange(len(sub)), columns = ['Folder','T1', 'T2', 'T3', 'SDRC', 'GMI', 'JIndex', 'after_eating'])
df_new['id'] = sub

for q in range(len(sub)):
#for q in range(10):
    T1 = []
    T2 = []
    T3 = []
    MG = []
    Src = []
    gmii = []
    ji = []
    af_e = []
    df_temp = time_glu_nonan[time_glu_nonan['Folder'] == sub[q]].reset_index(drop = True)
    for t in range(df_temp.shape[0]):
        T1.append(TR(df_temp.loc[t])[0])
        T2.append(TR(df_temp.loc[t])[1])
        T3.append(TR(df_temp.loc[t])[2])
        Src.append(SDRC(df_temp.loc[t]))
        gmii.append(GMI(df_temp.loc[t]))
        ji.append(Jindex(df_temp.loc[t]))
        af_e.append(slope_after_eating(df_temp.loc[t]))  
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
    df_new['SDRC'].loc[q] = np.mean(Src)
    df_new['GMI'].loc[q] = np.mean(gmii)
    df_new['JIndex'].loc[q] = np.mean(ji)
    df_new['after_eating'].loc[q] = np.nanmean(af_e)
    print("\r Process{}%".format(round((q+1)*100/len(sub))), end="")

df_new_outcome = pd.merge(outcome, df_new, on = 'Folder')
df_new_outcome = pd.get_dummies(df_new_outcome, columns=['Sex'], drop_first = True)

time_glu_nonan_new = time_glu_nonan[time_glu_nonan['Folder'].isin(df_new_outcome['Folder'])].reset_index(drop = True)
indexes_new = np.unique(time_glu_nonan_new['Folder'], return_index=True)[1]
sub_new = [time_glu_nonan_new['Folder'][index] for index in sorted(indexes_new)]

glu_num_new = []
for i in range(len(sub_new)):
    glu_num_new.append(time_glu_nonan_new[time_glu_nonan_new['Folder'] == sub_new[i]].shape[0])

glu_num_uni_new = np.unique(glu_num_new, return_counts=True)

plt.figure(figsize=(15,5))
plt.ylabel("Number of subjects")
plt.xlabel("Number of glucose measurements")
arr = plt.hist(glu_num_new, bins = 20)
for i in range(10):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))

X = preprocessing.scale(df_new_outcome[df_new_outcome.columns[6:-1]])
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
    plt.bar(['PC1', 'PC1 2', 'PC1 2 3', 'PC1 2 3 4'], cumvar, width = 1.0)
    plt.axhline(y=0.9, color='r', linestyle='-')

plotCumSumVariance(pca.explained_variance_ratio_)

def CV(p_grid, out_fold, in_fold, model, X, y, rand, n_beta = False):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        if (n_beta):
            beta.append(clf.best_estimator_.coef_[:n_beta])
        print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train)
        print(y_pred)
        r2train.append(r2_score(y_train, y_pred))
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        print(y_pred)
        r2test.append(r2_score(y_test, y_pred))
        
    if (n_beta):
        return r2train, r2test, beta
    else:
        return r2train, r2test

#Set parameters of cross-validation
par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
rand = 9

X_reg = np.concatenate((X, np.array(df_new_outcome[['Sex_Male']])), axis = 1)
BMI_train, BMI_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_new_outcome['bmi']), n_beta = False, rand = 66)
BMI_train, BMI_test, beta_BMI = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['bmi']), n_beta = 8, rand = 76)
MFratio_train, MFratio_test, beta_MFratio = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['Musclefatratio']), n_beta = 8, rand = 99)
FatPer_train, FatPer_test, beta_FatPer = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['Fat_percent']), n_beta = 8, rand = 99)
MuscleMass_train, MuscleMass_test, beta_MuscleMass = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['Muscle_mass']), n_beta = 8, rand = 99)
Fit_train, Fit_test, beta_Fit = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_new_outcome['FITNESS']), n_beta = 8, rand = 66)
Fit_train, Fit_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_new_outcome['FITNESS']), n_beta = False, rand = 9)

pd.DataFrame(beta_MFratio).T.to_csv('MFratio.csv')
pd.DataFrame(beta_FatPer).T.to_csv('FatPer.csv')
pd.DataFrame(beta_MuscleMass).T.to_csv('MuscleMass.csv')
pd.DataFrame(beta_Fit).T
model.fit(x_train, y_train)
model.predict(x_train)

-----------------------------------------------------------
df_new = pd.DataFrame(0, index = np.arange(len(df_12d_index)), columns = ['f1', 'f2'])
subs = [df_CGM_cpno['copsacno'][index] for index in df_12d_index]
df_new['copsacno'] = subs
frame_size = 6
hop_size = 3
window = 'hanning'

n = len(df_12d_index)
#features = np.empty((n, 2))

for i in range(n):
    df_temp = df_CGM_cpno[df_CGM_cpno['copsacno'] == sub[df_12d_index[i]]].reset_index(drop = True)['glucose_mmol_L']
    _, _, S = scipy.signal.stft(df_temp, nperseg=frame_size, noverlap=overlap_size)
    S = abs(S)

    # spectral centroid
    n_bins = len(S);
    bins = np.arange(n_bins).reshape(-1, 1)
    spectral_centroid = np.sum(bins*S, axis=0)/np.sum(S, axis=0)

    # spectral rolloff
    cum_energy = np.cumsum(S, axis=0);
    cum_energy_percent = cum_energy/np.sum(S, axis=0);
    spectral_rolloff = np.argmax(cum_energy_percent > 0.85, axis=0)

    # add to feature matrix
    df_new['f1'].loc[i] = spectral_centroid.mean()
    df_new['f2'].loc[i] = spectral_rolloff.mean()
    print("\r Process{}%".format(round((i+1)*100/len(df_12d_index))), end="")



for q in range(len(df_12d_index)):
    f1 = []
    f2 = []
    for t in range(0, 1151, 96):
        df_temp = df_CGM_cpno[df_CGM_cpno['copsacno'] == sub[df_12d_index[q]]].reset_index(drop = True).loc[t:t+95].reset_index(drop = True)['glucose_mmol_L']
        _, _, S = scipy.signal.stft(df_temp, nperseg=frame_size, noverlap=overlap_size)
        S = abs(S)

        # spectral centroid
        n_bins = len(S);
        bins = np.arange(n_bins).reshape(-1, 1)
        spectral_centroid = np.sum(bins*S, axis=0)/np.sum(S, axis=0)

        # spectral rolloff
        cum_energy = np.cumsum(S, axis=0);
        cum_energy_percent = cum_energy/np.sum(S, axis=0);
        spectral_rolloff = np.argmax(cum_energy_percent > 0.85, axis=0)

        # add to feature matrix
        f1.append(spectral_centroid.mean())
        f2.append(spectral_rolloff.mean())
    df_new['f1'].loc[q] = np.mean(f1)
    df_new['f2'].loc[q] = np.mean(f2)
    print("\r Process{}%".format(round((q+1)*100/len(df_12d_index))), end="")

# stft parameters
frame_size = 2
hop_size = 1
window = 'hanning'

e = '303'
#e = '222'

df_CGM_sub = df_12d[df_12d['copsacno'] == e]['glucose_mmol_L'].reset_index(drop = True)
#df_temp = df1['glucose_mmol_L']
#df_CGM_sub = df_temp
fs=len(df_CGM_sub)/14

k = Non_all_inf.index(e)
df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == Non_all_inf[k]].reset_index(drop = True)
eating = []
eating_index = []
for j in range(1, df1.shape[0]):
    if (df1['TimeSinceLastPicture'].loc[j] < df1['TimeSinceLastPicture'].loc[j - 1]):
        eating_index.append(j)
        eating.append(df1['Time'].loc[j])  
mark = [range(df1['Time'].shape[0]).index(i) for i in eating_index]

# plot
fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
t = np.arange(len(df_CGM_sub))/fs
#axes[0].plot(t, df_CGM_sub)
#axes[0].set_xlabel('Time (s)')
#axes[0].set_ylabel('Amplitude')
axes[0].scatter(t, df_CGM_sub, marker = '.')      
axes[0].plot(t[eating_index],[df1['glucose_mmol_L'][i] for i in mark], ls="", marker="o", label="points", c = 'orange')
overlap_size = frame_size - hop_size
f, t, S = scipy.signal.stft(df_CGM_sub, fs=fs, nperseg=frame_size, noverlap=overlap_size)
S_dB = 20*np.log10(abs(S))
axes[1].pcolormesh(t, f, S_dB, shading='auto')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Frequency')
fig.tight_layout()
plt.show()

def plot_indi(k, eatdot = True, resolu = 1000):
    df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == Non_all_inf[k]].reset_index(drop = True)
    #df1 = df_12d_non_all_inf[df_12d_non_all_inf['copsacno'] == k].reset_index(drop = True)
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

    plt.figure(figsize=(20,5), dpi = resolu)
    plt.rcParams.update({'font.size': 20})

    #colors = ['purple','blue']
    #plt.scatter(df1['Time'], df1['glucose_mmol_L'], marker = '.', c = pd.get_dummies(df1['is_sleep'])['awake'].astype(int).to_list(), cmap = matplotlib.colors.ListedColormap(colors))
    plt.scatter(df1['Time'], df1['glucose_mmol_L'], marker = '.')
    
    mark = [range(df1['Time'].shape[0]).index(i) for i in eating_index]

    if (eatdot):
        plt.plot(eating,[df1['glucose_mmol_L'][i] for i in mark], ls="", marker="o", label="points", c = 'orange')

for q in range(len(df_12d_index)):
    print(q)
#for q in range(10):
    for t in range(0, 1151, 96):
        df_temp = df_CGM_cpno[df_CGM_cpno['copsacno'] == sub[df_12d_index[q]]].reset_index(drop = True).loc[t:t+95].reset_index(drop = True)
        T1.append(TR(df_temp)[0])
        T2.append(TR(df_temp)[1])
        T3.append(TR(df_temp)[2])
        MG.append(np.mean(df_temp['glucose_mmol_L']))
        Src.append(SDRC(df_temp))
        gmii.append(GMI(df_temp))
        ji.append(Jindex(df_temp)) 
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
    df_new['SDRC'].loc[q] = np.mean(Src)
    df_new['GMI'].loc[q] = np.mean(gmii)
    df_new['JIndex'].loc[q] = np.mean(ji)
    print("\r Process{}%".format(round((q+1)*100/len(df_12d_index))), end="")


df_pheno = pd.read_csv('~/data/CGM/CGM_DATA/CGM3.csv')
df_pheno['copsacno'] = df_pheno['copsacno'].astype(str)
df_pheno = df_pheno.drop_duplicates(subset="copsacno")

df_ft_pheno_ft = pd.merge(df_new, df_pheno, on = 'copsacno')

df_ft_pheno = pd.merge(df_ft, df_pheno, on = 'copsacno')
df_ft_pheno = df_ft_pheno[df_ft_pheno['Musclefatratio'].notna()]
df_ft_pheno = df_ft_pheno[df_ft_pheno['Muscle_mass'].notna()]
df_ft_pheno = df_ft_pheno[df_ft_pheno['Bone_mass'].notna()]

df_ft_pheno_ftmodel = pd.merge(df_ft_pheno, model_results, on = 'copsacno')

scaler = StandardScaler()
#x_var = ['T1', 'T2', 'T3', 'SDRC', 'GMI', 'JIndex']
x_var = ['p1 median', 'p2 median', 'SI median', 'p1 CV', 'p2 CV',
       'SI CV', 'householdincome', 'm_edu', 'Mother_age_2yrs']
x_var = df_ft.columns[:-1]
scaler.fit(np.array(df_ft_pheno[x_var]))
X = scaler.transform(np.array(df_ft_pheno[x_var])) 
X = np.array(df_ft_pheno[x_var])
X1 = np.array(df_ft_pheno[['p1 median', 'p2 median', 'SI median', 'p1 CV', 'p2 CV', 'SI CV']])
X_reg = np.concatenate((X, X1, np.array(df_ft_pheno[['Sex_binary']])), axis = 1)
#X_reg = np.concatenate((np.array(df_model_pheno[['Sex_binary']]), np.array(df_model_pheno[['Race_non-caucasian']])), axis = 1)
X_reg = X
#BMI_train, BMI_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_new_outcome['bmi18y']), n_beta = False, rand = 66)
BMI_train, BMI_test, beta_BMI = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['bmi18y']), n_beta = 11, rand = 76)
MFratio_train, MFratio_test, beta_MFratio = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['Musclefatratio']), n_beta = 8, rand = 99)
Fit_train, Fit_test, beta_Fit = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['FITNESS']), n_beta = 11, rand = 66)
MuscleMass_train, MuscleMass_test, beta_MuscleMass = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['Muscle_mass']), n_beta = 11, rand = 99)
WHR_train, WHR_test, beta_WHR = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['waisthipratio']), n_beta = 11, rand = 99)
Bonemass_train, Bonemass_test, beta_Bonemass = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['Bone_mass']), n_beta = 11, rand = 99)
Tfat_train, Tfat_test, beta_Tfat = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['Trunkfat_percent']), n_beta = 11, rand = 99)
Sk_train, Sk_test, Sk_Vfat = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['Skelmuscle_mass']), n_beta = 11, rand = 99)

X1 = np.array(df_model_pheno_model[['p1 median', 'p2 median', 'SI median', 'p1 CV', 'p2 CV', 'SI CV']])
X_reg = np.concatenate((X1, np.array(df_model_pheno_model[['Sex_binary']])), axis = 1)

MFratio_train_model, MFratio_test_model, beta_MFratio_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno_model['Musclefatratio']), n_beta = 8, rand = 99)
MuscleMass_train_model, MuscleMass_test_model, beta_MuscleMass_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno_model['Muscle_mass']), n_beta = 11, rand = 99)
Bonemass_train_model, Bonemass_test_model, beta_Bonemass_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno_model['Bone_mass']), n_beta = 11, rand = 99)

x_var = df_ft.columns[:-1]
scaler.fit(np.array(df_ft_pheno[x_var]))
X = scaler.transform(np.array(df_ft_pheno[x_var])) 
X_reg = np.concatenate((X, np.array(df_ft_pheno[['Sex_binary']])), axis = 1)

MFratio_train_ft, MFratio_test_ft, beta_MFratio_ft = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['Musclefatratio']), n_beta = 8, rand = 99)
MuscleMass_train_ft, MuscleMass_test_ft, beta_MuscleMass_ft = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['Muscle_mass']), n_beta = 11, rand = 99)
Bonemass_train_ft, Bonemass_test_ft, beta_Bonemass_ft = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno['Bone_mass']), n_beta = 11, rand = 99)

x_var1 = ['p1 median', 'p2 median', 'SI median', 'p1 CV', 'p2 CV',
       'SI CV', 'householdincome', 'm_edu', 'Mother_age_2yrs']
x_var2 = df_ft.columns[:-1]
scaler.fit(np.array(df_ft_pheno_ftmodel[x_var2]))
X = scaler.transform(np.array(df_ft_pheno_ftmodel[x_var2])) 
#X = np.array(df_ft_pheno_ftmodel[x_var])
X1 = np.array(df_ft_pheno_ftmodel[['p1 median', 'p2 median', 'SI median', 'p1 CV', 'p2 CV', 'SI CV']])
X_reg = np.concatenate((X, X1, np.array(df_ft_pheno_ftmodel[['Sex_binary']])), axis = 1)

MFratio_train_ftmodel, MFratio_test_ftmodel, beta_MFratio_ftmodel = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno_ftmodel['Musclefatratio']), n_beta = 8, rand = 99)
MuscleMass_train_ftmodel, MuscleMass_test_ftmodel, beta_MuscleMass_ftmodel = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno_ftmodel['Muscle_mass']), n_beta = 11, rand = 99)
Bonemass_train_ftmodel, Bonemass_test_ftmodel, beta_Bonemass_ftmodel = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_ft_pheno_ftmodel['Bone_mass']), n_beta = 11, rand = 99)

labels = ['Dynamic model parameters', 'Time-series features', 'Both']
dpi = 900
title = 'Regression results of outcomes and CGM features'

x = np.arange(len(labels))

mfr_mean = [np.mean(MFratio_test_model), np.mean(MFratio_test_ftmodel), np.mean(MuscleMass_test_ft)]
b_mean = [np.mean(Bonemass_test_model), np.mean(Bonemass_test_ft), np.mean(Bonemass_test_ftmodel)]
M_mean = [np.mean(MuscleMass_test_model), np.mean(MuscleMass_test_ft), np.mean(MuscleMass_test_ftmodel)]
mfr_std = [np.std(MFratio_test_model), np.std(MFratio_test_ftmodel), np.std(MuscleMass_test_ft)]
b_std = [np.std(Bonemass_test_model), np.std(Bonemass_test_ft), np.std(Bonemass_test_ftmodel)]   
M_std = [np.std(MuscleMass_test_model), np.std(MuscleMass_test_ft), np.std(MuscleMass_test_ftmodel)]

#train_mean = [np.mean(MFratio_train), np.mean(Bonemass_train), np.mean(MuscleMass_train)]
#test_mean = [np.mean(MFratio_test), np.mean(Bonemass_test), np.mean(MuscleMass_test)]
#train_std = [np.std(MFratio_train),  np.std(Bonemass_train), np.std(MuscleMass_train)]
#test_std = [np.std(MFratio_test), np.std(Bonemass_test), np.std(MuscleMass_test)]    

#train_mean = [np.mean(Sk_train), np.mean(MFratio_train), np.mean(Bonemass_train), np.mean(MuscleMass_train)]
#test_mean = [np.mean(Sk_test), np.mean(MFratio_test), np.mean(Bonemass_test), np.mean(MuscleMass_test)]
#train_std = [np.std(Sk_train), np.std(MFratio_train),  np.std(Bonemass_train), np.std(MuscleMass_train)]
#test_std = [np.std(Sk_test), np.std(MFratio_test), np.std(Bonemass_test), np.std(MuscleMass_test)]    

fig, ax = plt.subplots(dpi = 1000)

width = 0.4
rects1 = ax.bar(x - width/2, mfr_mean, width/2, yerr = mfr_std, label='Muscle-fat ratio', align='center', ecolor='black', capsize=2, color = 'dodgerblue')
rects2 = ax.bar(x, b_mean, width/2, yerr = b_std, label='Bone mass', align='center', ecolor='black', capsize=2)
rects3 = ax.bar(x + width/2, M_mean, width/2, yerr = M_std, label='Muscle mass', align='center', ecolor='black', capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2')
ax.set_title(title, fontsize = 7)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize = 5)
ax.legend(loc = 'upper right', bbox_to_anchor=(1.05,1.), fontsize = 6)
#plt.xticks(rotation=60)
#plt.yticks(np.linspace(0.0,0.7,0.1))
ax.set_xticklabels(labels, fontsize = 6)
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
#for i, v in enumerate(train_mean):
   # rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
#for i, v in enumerate(test_mean):
    #rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
fig.tight_layout()

plt.show()   

from scipy.fftpack import fft, ifft

def fft_and_peaks(df):
    n = len(df.columns)
    fft_cols= []
    for i in range(n):
        if ('y' in df.columns[i])==True:
            fft_cols.append(df.columns[i])
        else:
            pass
        
    fft_val=[]
    for j in fft_cols:
        fft_val.append(abs(fft(df[j])))
        
    peak_val=[]
    for z in range(len(fft_val)):
        a = list(set(fft_val[z]))
        a.sort()
        a = a[::-1][1:5]
        peak_val.append(a)
    return(fft_val,peak_val)
fft_val_pat1,peak_val_pat1 = fft_and_peaks(df_temp)
fig = px.line(y=fft_val_pat1[0], x=range(new_data_pat1.shape[0]),labels={'x':'Frequency','y':'Amplitude'},title='Plot of Amplitude vs Frequency for Instance 0')
fig.show()

def ft_feature(df_temp):
    fft_val = abs(fft(df_temp.values))
    a = list(set(fft_val))
    a.sort()
    a = a[::-1][1:5]

    v = []
    for i in range(len(df_temp.values)):
        if i == 0:
            v.append(0)
        else:
            v.append((abs(df_temp.values[i]-df_temp.values[i-1]))/15)
    v_f = np.array(v)[np.argpartition(v, -4)[-4:]]

    auto_corr = np.array([df_temp.autocorr(lag=2), df_temp.autocorr(lag=3), df_temp.autocorr(lag=4), df_temp.autocorr(lag=5), df_temp.autocorr(lag=6), df_temp.autocorr(lag=7), df_temp.autocorr(lag=8), df_temp.autocorr(lag=9)])

    y = df_temp.to_numpy()
    time = np.linspace(1,len(df_temp),len(df_temp))
    coeff = np.polyfit(time,y,6)
    return np.concatenate((np.array(a), v_f, auto_corr, coeff), axis = 0)

df_new = pd.DataFrame(0, index = np.arange(len(df_12d_index)), columns = ['f1', 'f2'])
subs = [sub[i] for i in df_12d_index]
df_new['copsacno'] = subs

ft_features = []
for q in range(len(df_12d_index)):
    ft_t = []
    for t in range(0, 1151, 96):
        df_temp = df_CGM_cpno[df_CGM_cpno['copsacno'] == sub[df_12d_index[q]]].reset_index(drop = True).loc[t:t+95].reset_index(drop = True)['glucose_mmol_L']
        ft_t.append(ft_feature(df_temp)) 
        
    ft_features.append(np.array(ft_t).mean(axis = 0))
    print("\r Process{}%".format(round((q+1)*100/len(df_12d_index))), end="")

df_ft = pd.DataFrame(ft_features)
subs = [sub[i] for i in df_12d_index]
df_ft['copsacno'] = subs
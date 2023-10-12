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

def getIndexes(dfObj, value):
     
    # Empty list
    listOfPos = []
     
    # isin() method will return a dataframe with
    # boolean values, True at the positions   
    # where element exists
    result = dfObj.isin([value])
     
    # any() method will return
    # a boolean series
    seriesObj = result.any()
 
    # Get list of column names where
    # element exists
    columnNames = list(seriesObj[seriesObj == True].index)
    
    # Iterate over the list of columns and
    # extract the row index where element exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
 
        for row in rows:
            listOfPos.append((row, col))
             
    # This list contains a list tuples with
    # the index of element in the dataframe
    return listOfPos

df_diet = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/NAS_data/Diet_chanllenge_BiochemData.csv')
df_diet = df_diet.rename(columns = {"COPSACNO":"copsacno"})
df_diet_noheader = df_diet.tail(df_diet.shape[0] - 4)
df_diet_noheader['copsacno'] = df_diet_noheader['copsacno'].astype(str)
df_diet_noheader.reset_index(drop = True, inplace = True)

df_diet_noheader = df_diet_noheader.drop(df_diet_noheader.columns[0:3], axis = 1)
df_diet_noheader = df_diet_noheader.drop(df_diet_noheader.columns[[2,3]], axis = 1)
df_diet_noheader_nonan = df_diet_noheader.drop(['Phe', 'XXL-VLDL-PL %', 'XXL-VLDL-C %', 'XXL-VLDL-CE %', 'XXL-VLDL-FC %', 'XXL-VLDL-TG %', 'Glycerol'], axis = 1)
df_diet_noheader_nonan = df_diet_noheader_nonan.dropna()
df_diet_noheader_nonan = df_diet_noheader_nonan[~df_diet_noheader_nonan.isin(['TAG']).any(axis = 1)]
df_diet_noheader_nonan = df_diet_noheader_nonan.groupby('copsacno').filter(lambda x: len(x) == 8)
df_diet_noheader_nonan = df_diet_noheader_nonan.drop_duplicates(subset=['copsacno','TIMEPOINT'])
df_diet_noheader_nonan = df_diet_noheader_nonan.sort_values(['copsacno', 'TIMEPOINT'])
df_diet_noheader_nonan.reset_index(drop = True, inplace = True)
df_diet_noheader_nonan_pivot = df_diet_noheader_nonan.pivot(index='copsacno', columns='TIMEPOINT')

df_glu_insulin_Cpep = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df_glu_insulin_Cpeptid.csv')
df_glu_insulin_Cpep['copsacno'] = df_glu_insulin_Cpep['copsacno'].astype(str)
df_glu_insulin_Cpep_nonan = df_glu_insulin_Cpep.groupby('copsacno').filter(lambda x: len(x) == 8)
df_glu_insulin_Cpep_nonan = df_glu_insulin_Cpep_nonan.drop(df_glu_insulin_Cpep_nonan[df_glu_insulin_Cpep_nonan['copsacno'] == '118'].index)
df_glu_insulin_nonan = df_glu_insulin_Cpep_nonan.drop(['age'], axis=1)
df_glu_insulin_nonan_pivot = df_glu_insulin_nonan.pivot(index='copsacno', columns='TIMEPOINT')

df_pheno_all = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df2000.csv')
df_pheno_all = df_pheno_all.rename(columns = {"COPSACNO":"copsacno"})
df_pheno_all['copsacno'] = df_pheno_all['copsacno'].astype(str)
df_pheno_all_BMI = df_pheno_all[['copsacno', 'Sex', 'BMI']].dropna(subset = ['BMI'])
df_pheno_all_BMI = df_pheno_all[['copsacno', 'Sex', 'Musclefatratio']].dropna(subset = ['Musclefatratio'])

df_BMI_gluT_nonnan = pd.merge(df_pheno_all_BMI, df_glu_insulin_nonan_pivot, on = 'copsacno')
df_BMI_gluT_nonnan.columns = ['{}_{}'.format(x[0], x[1]) for x in df_BMI_gluT_nonnan.columns]
df_BMI_gluT_nonnan.rename(columns = {df_BMI_gluT_nonnan.columns[0]: 'copsacno', df_BMI_gluT_nonnan.columns[1]: 'Sex', df_BMI_gluT_nonnan.columns[2]: 'Musclefatratio'},inplace=True)
df_BMI_gluT_nonnan.reset_index(drop = True, inplace = True)

df_BMI_gluT_nonnan_OMMG = pd.merge(df_BMI_gluT_nonnan, model_results, on = 'copsacno')
df_BMI_gluT_nonnan = df_BMI_gluT_nonnan_OMMG

df_BMI_diet_nonnan = pd.merge(df_pheno_all_BMI, df_diet_noheader_nonan_pivot, on = 'copsacno')
df_BMI_diet_nonnan.columns = ['{}_{}'.format(x[0], x[1]) for x in df_BMI_diet_nonnan.columns]
df_BMI_diet_nonnan.rename(columns = {df_BMI_diet_nonnan.columns[0]: 'copsacno', df_BMI_diet_nonnan.columns[1]: 'Sex', df_BMI_diet_nonnan.columns[2]: 'Musclefatratio'},inplace=True)
df_BMI_diet_nonnan.reset_index(drop = True, inplace = True)

df_all = pd.merge(df_BMI_gluT_nonnan, df_BMI_diet_nonnan, on = ['copsacno', 'Sex', 'Musclefatratio'])
#df_BMI_gluT_nonnan = df_BMI_diet_nonnan

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
par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
rand = 41

scaler1 = StandardScaler()
x_var1 = df_BMI_gluT_nonnan.columns[3:]
scaler1.fit(np.array(df_BMI_gluT_nonnan[x_var1]))
X1 = scaler1.transform(np.array(df_BMI_gluT_nonnan[x_var1])) 

X_reg1 = np.concatenate((X1, np.array(df_BMI_gluT_nonnan[['Sex']])), axis = 1)
#X_reg1 = X1
BMI_train_SVR, BMI_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg1, y = preprocessing.scale(df_BMI_gluT_nonnan['Musclefatratio']), n_beta = False, rand = rand)
BMI_train_E, BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 2000, tol = 5e-3), X = X_reg1, y = preprocessing.scale(df_BMI_gluT_nonnan['Musclefatratio']), n_beta = False, rand = rand)

np.mean(BMI_test_SVR)
np.mean(BMI_test_E)

BMI_train_SVR, BMI_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = np.log(df_BMI_gluT_nonnan['BMI']), n_beta = False, rand = rand)
BMI_train_E, BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 2000, tol = 2e-3), X = X_reg, y = np.log(df_BMI_gluT_nonnan['BMI']), n_beta = False, rand = rand)


#BMI_train, BMI_test, beta_BMI = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_BMI_diet_nonnan['BMI']), n_beta = False, rand = rand)

labels = ['SVR', 'Elastic-net']
dpi = 600
title = 'Regression results of logBMI'

x = np.arange(len(labels))
train_mean = [np.mean(BMI_train_SVR), np.mean(BMI_train_E)]
test_mean = [np.mean(BMI_test_SVR), np.mean(BMI_test_E)]
train_std = [np.std(BMI_train_SVR),  np.std(BMI_train_E)]
test_std = [np.std(BMI_test_SVR), np.std(BMI_test_E)]    

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
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
#for i, v in enumerate(train_mean):
   # rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
#for i, v in enumerate(test_mean):
    #rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
fig.tight_layout()

plt.show()   
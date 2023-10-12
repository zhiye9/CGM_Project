import numpy as np
import sys
sys.path.append('/home/zhi/data/CGM/VBA_CP/VBA-OMM/Python/')
import VBA_OMM
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

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
from sklearn.metrics import mean_absolute_error
import warnings
import time
# Read the demo data from csv file
#df = pd.read_csv("DataOMM_G.csv")
df_glu_insulin_Cpep = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df_glu_insulin_Cpeptid.csv')
df_glu_insulin_Cpep['copsacno'] = df_glu_insulin_Cpep['copsacno'].astype(str)
df_glu_insulin_Cpep_nonan = df_glu_insulin_Cpep.groupby('copsacno').filter(lambda x: len(x) == 8)
#df = df_glu_insulin_Cpep[df_glu_insulin_Cpep['copsacno'] == 101]

def VBA_glu(test, method = 'RaPL', display = False):
    t = test["TIMEPOINT"].to_numpy()
    G = test["Glucose"].to_numpy()
    I = test["Insulin"].to_numpy()

    # Construt the data struture
    dat = {"t": t.reshape(len(t),1),
        "G": G.reshape(len(t),1),
        "I": I.reshape(len(t),1)}

    # Constants
    const = {"A": 6,                    
            "V": 0.145,                # L/kg
            "dt": 0.1,                 # min
            "Rap": [],                 # mmol/kg/min
            "X0": 0,                   # 1/min
            "measCV": 2,               # %
            "Gb": dat["G"][0, 0],      # mmol/L
            "G0": dat["G"][0, 0],      # mmol/L
            "Ib": dat["I"][0, 0]}      # pmol/L

    # Construct inversion options
    opt = {"GA_fun": "RaPL",
        "tb": np.array([0, 15, 30, 60, 90, 120, 150, 240]),      # min
        "alpha": 0.017,                                          # 1/min
        "displayWin": display}

    # Priors
        # - System Parameters [median CV]
    priors = {"p1": np.array([0.025, 25]),      # 1/min
            "p2": np.array([0.012, 40]),      # 1/min
            "SI": np.array([12E-5, 100])}     # 1/min per pmol/L

        # - Input function Parameters
    if opt["GA_fun"] == 'RaPL':
        priors.update({"k": np.array([[3.2E-3*const["A"], 50],          # mmol/kg/min
                                    [7.3E-3*const["A"], 50],          # mmol/kg/min
                                    [5.4E-3*const["A"], 50],          # mmol/kg/min
                                    [5.1E-3*const["A"], 50],          # mmol/kg/min
                                    [3.7E-3*const["A"], 50],          # mmol/kg/min    
                                    [1.8E-3*const["A"], 50]])})       # mmol/kg/min
    if opt["GA_fun"] == 'RaLN':
        priors.update({"k": np.array([[30, 30],         # min
                                    [0.5, 30],        
                                    [100, 30],        # min
                                    [0.5, 30],
                                    [0.7, 30]])})

    return VBA_OMM.mainG(dat, priors, const, opt)


p1_median = []
p2_median = []
SI_median = []

p1_CV = []
p2_CV = []
SI_CV = []

out_list = []
R2 = []
RMSE = []
subs = []

subjects = np.unique(df_glu_insulin_Cpep_nonan['copsacno'])

for i in range(len(subjects)):
    test = df_glu_insulin_Cpep[df_glu_insulin_Cpep['copsacno'] == subjects[i]].sort_values('TIMEPOINT')[['copsacno', 'TIMEPOINT', 'Glucose', 'Insulin']]
    subs.append(subjects[i])
    out = VBA_glu(test, method = 'RaLN')
    #out_list.append(out)
    p1_median.append(out['posterior']['p1'][0])
    p2_median.append(out['posterior']['p2'][0])
    SI_median.append(out['posterior']['SI'][0])
    #p1_CV.append(out['posterior']['p1'][1])
    #p2_CV.append(out['posterior']['p2'][1])
    #SI_CV.append(out['posterior']['SI'][1])
    #R2.append(out['Performance']['R2'])
    #RMSE.append(out['Performance']['RMSE'])

    print("\r Process{}%".format(round((i+1)*100/len(subjects))), end="")


os.chdir('/home/zhi/data/CGM/CGM_results/OMM_G_new')
#ith open("out_list", "wb") as fp:   
 #   pickle.dump(out_list, fp)
#with open("out_list2", "wb") as fp:   
 #   pickle.dump(out_list, fp)

#with open("out_list1", "rb") as fp:
 #   out_list_PL = pickle.load(fp)

#with open("out_list2", "rb") as fp:
    #out_list_LN = pickle.load(fp)

with open("p1_median_new", "wb") as fp:   
    pickle.dump(p1_median, fp)
with open("p1_median", "rb") as fp:
    p1_median = pickle.load(fp)

with open("p2_median_new", "wb") as fp:   
    pickle.dump(p2_median, fp)
with open("p2_median", "rb") as fp:
    p2_median = pickle.load(fp)

with open("SI_median_new", "wb") as fp:   
    pickle.dump(SI_median, fp)
with open("SI_median", "rb") as fp:
    SI_median = pickle.load(fp)

#with open("p1_CV", "wb") as fp:   
 #   pickle.dump(p1_CV, fp)
with open("p1_CV", "rb") as fp:
    p1_CV = pickle.load(fp)

#with open("p2_CV", "wb") as fp:   
 #   pickle.dump(p2_CV, fp)
with open("p2_CV", "rb") as fp:
    p2_CV = pickle.load(fp)

#with open("SI_CV", "wb") as fp:   
 #   pickle.dump(SI_CV, fp)
with open("SI_CV", "rb") as fp:
    SI_CV = pickle.load(fp)

with open("R2", "wb") as fp:   
    pickle.dump(R2, fp)

with open("RMSE", "wb") as fp:   
    pickle.dump(RMSE, fp)


model_results = pd.DataFrame({'copsacno': subs,
                                'p1 median': preprocessing.scale(p1_median), 
                        'p2 median': preprocessing.scale(p2_median),
                        'SI median': preprocessing.scale(SI_median),
                        #'p1 CV': preprocessing.scale(p1_CV), 
                        #'p2 CV': preprocessing.scale(p2_CV),
                        #'SI CV': preprocessing.scale(SI_CV),
                        #'R2': R2,
                        #'RMSE': RMSE
                        })

model_results = pd.DataFrame({'copsacno': subs,
                                'p1 median': p1_median, 
                        'p2 median': p2_median,
                        'SI median': SI_median,
                         })
           #             'p1 CV': p1_CV, 
             #           'p2 CV': p2_CV,
              #          'SI CV': SI_CV,
                        #'R2': R2,
                        #'RMSE': RMSE
                        })


df_pheno_all = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df2000.csv')
df_pheno_all = df_pheno_all.rename(columns = {"COPSACNO":"copsacno"})
df_pheno_all['copsacno'] = df_pheno_all['copsacno'].astype(str)
df_pheno_all_BMI = df_pheno_all[['copsacno', 'Sex', 'BMI']].dropna(subset = ['BMI'])
df_pheno_all_BMI = df_pheno_all[['copsacno', 'Sex', 'Musclefatratio']].dropna(subset = ['Musclefatratio'])

df_BMI_OMMG_nonnan = pd.merge(df_pheno_all_BMI, model_results, on = 'copsacno')
------------
df_pheno_all_BMI = df_pheno_all[['copsacno', 'Sex', 'BMI', 'Muscle_mass']].dropna(subset = ['BMI', 'Muscle_mass'])
df_BMI_OMMG_nonnan = pd.merge(df_pheno_all_BMI, model_results, on = 'copsacno')
np.corrcoef(df_BMI_OMMG_nonnan['BMI'], df_BMI_OMMG_nonnan['Muscle_mass'])
------------
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
        #r2train.append(mean_absolute_error(y_train, y_pred))

        #predict labels on the test set
        y_pred = clf.predict(x_test)
        #print(y_pred)
        r2test.append(r2_score(y_test, y_pred))
        #r2test.append(mean_absolute_error(y_test, y_pred))
        
    if (n_beta):
        return r2train, r2test, beta
    else:
        return r2train, r2test


#Set parameters of cross-validation
par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
rand = 49

scaler = StandardScaler()
x_var = df_BMI_OMMG_nonnan.columns[3:]
#X = np.array(df_BMI_OMMG_nonnan[x_var])
scaler.fit(np.array(df_BMI_OMMG_nonnan[x_var]))
X = scaler.transform(np.array(df_BMI_OMMG_nonnan[x_var])) 

X_reg = np.concatenate((X, np.array(df_BMI_OMMG_nonnan[['Sex']])), axis = 1)
#X_reg = X
BMIglu_train_SVR, BMIglu_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_BMI_OMMG_nonnan['Musclefatratio']), n_beta = False, rand = rand)
BMIglu_train_E, BMIglu_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_BMI_OMMG_nonnan['Musclefatratio']), n_beta = False, rand = rand)

np.mean(BMIglu_test_SVR)
np.mean([i for i in BMIglu_test_E if i > 0])

R2 = [i['Performance']['R2'] for i in out_list]
RMSE = [i['Performance']['RMSE'] for i in out_list]

fig, ax = plt.subplots(dpi = 1000)
ax.set_title('Distribution of R2 for each subject')
plt.hist(R2, bins = 100)

fig, ax = plt.subplots(dpi = 1000)
ax.set_title('Distribution of RMSE for each subject')
plt.hist(RMSE, bins = 100)

df_BMI_OMMG_nonnan = df_BMI_OMMG_nonnan[['p1 median', 'p2 median', 'SI median', 'Musclefatratio', 'Sex']]
df_BMI_OMMG_m = df_BMI_OMMG_nonnan[df_BMI_OMMG_nonnan['Sex'] == 1][['p1 median', 'p2 median', 'SI median', 'Musclefatratio', 'Sex']]
df_BMI_OMMG_f = df_BMI_OMMG_nonnan[df_BMI_OMMG_nonnan['Sex'] == 0][['p1 median', 'p2 median', 'SI median', 'Musclefatratio', 'Sex']]
df_BMI_OMMG_nonnan = df_BMI_OMMG_nonnan[['p1 median', 'p2 median', 'SI median', 'Sex']]
df_BMI_OMMG_m = df_BMI_OMMG_nonnan[df_BMI_OMMG_nonnan['Sex'] == 1][['p1 median', 'p2 median', 'SI median', 'Sex']]
df_BMI_OMMG_f = df_BMI_OMMG_nonnan[df_BMI_OMMG_nonnan['Sex'] == 0][['p1 median', 'p2 median', 'SI median', 'Sex']]

df_BMI_OMMG_nonnan = df_BMI_OMMGCP_nonnan[['T_median', 'beta_median', 'kd_median', 'Sex']]
df_BMI_OMMG_m = df_BMI_OMMG_nonnan[df_BMI_OMMG_nonnan['Sex'] == 1][['T_median', 'beta_median', 'kd_median', 'Sex']]
df_BMI_OMMG_f = df_BMI_OMMG_nonnan[df_BMI_OMMG_nonnan['Sex'] == 0][['T_median', 'beta_median', 'kd_median', 'Sex']]

df_BMI_OMMG_nonnan_log = df_BMI_OMMG_nonnan
df_BMI_OMMG_nonnan_log['T_median'] = np.log(df_BMI_OMMG_nonnan['T_median'])
df_BMI_OMMG_nonnan_log['beta_median'] = np.log(df_BMI_OMMG_nonnan['beta_median'])
df_BMI_OMMG_nonnan_log['kd_median'] = np.log(df_BMI_OMMG_nonnan['kd_median'])
df_BMI_OMMG_nonnan_log = df_BMI_OMMG_nonnan_log[['T_median', 'beta_median', 'kd_median', 'Sex']]
df_BMI_OMMG_m = df_BMI_OMMG_nonnan_log[df_BMI_OMMG_nonnan_log['Sex'] == 1][['T_median', 'beta_median', 'kd_median', 'Sex']]
df_BMI_OMMG_f = df_BMI_OMMG_nonnan_log[df_BMI_OMMG_nonnan_log['Sex'] == 0][['T_median', 'beta_median', 'kd_median', 'Sex']]


df_BMI_OMMGCP_nonnan_log = df_BMI_OMMGCP_nonnan[['p1 median', 'p2 median', 'SI median', 'Sex']]
df_BMI_OMMGCP_nonnan_log['p1 median'] = preprocessing.scale(df_BMI_OMMGCP_nonnan_log['p1 median'])
df_BMI_OMMGCP_nonnan_log['p2 median'] = preprocessing.scale(df_BMI_OMMGCP_nonnan_log['p2 median'])
df_BMI_OMMGCP_nonnan_log['SI median'] = preprocessing.scale(df_BMI_OMMGCP_nonnan_log['SI median'])
df_BMI_OMMG_m = df_BMI_OMMGCP_nonnan_log[df_BMI_OMMGCP_nonnan_log['Sex'] == 1][['p1 median', 'p2 median', 'SI median', 'Sex']]
df_BMI_OMMG_f = df_BMI_OMMGCP_nonnan_log[df_BMI_OMMGCP_nonnan_log['Sex'] == 0][['p1 median', 'p2 median', 'SI median',  'Sex']]

df_long = pd.melt(df_BMI_OMMGCP_nonnan_log, "Sex", var_name="a", value_name="c")
g = sns.factorplot("a", hue="Sex", y="c", data=df_long, kind="box")
legend = g._legend

legend.set_title("Sex")
for t, l in zip(legend.texts,("Male", "Female")):
    t.set_text(l)

ax = sns.boxplot(data=df_BMI_OMMG_m, x = df_BMI_OMMG_m[:3], hue = "Sex")
 
# Add jitter with the swarmplot function
#ax = sns.swarmplot(data=df_BMI_OMMG_m['T_median', 'beta_median', 'kd_median'], hue = "Sex")
ax = sns.swarmplot(data=df_BMI_OMMG_m['p1 median', 'p2 median', 'SI median'], hue = "Sex")
plt.show()
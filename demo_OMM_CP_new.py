import numpy as np
import sys
sys.path.append('/home/zhi/data/CGM/VBA_CP/VBA-OMM/Python/')
import VBA_OMM
import pandas as pd
import pickle
import seaborn as sns

# Read the demo data from csv file
#df = pd.read_csv("/home/zhi/data/CGM/VBA_CP/VBA-OMM/Python/Data1.csv")
df_glu_insulin_Cpep = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df_glu_insulin_Cpeptid.csv')
df_glu_insulin_Cpep['copsacno'] = df_glu_insulin_Cpep['copsacno'].astype(str)
df_glu_insulin_Cpep_nonan = df_glu_insulin_Cpep.groupby('copsacno').filter(lambda x: len(x) == 8)
df_glu_insulin_Cpep_nonan = df_glu_insulin_Cpep_nonan.drop(df_glu_insulin_Cpep_nonan[df_glu_insulin_Cpep_nonan['copsacno'] == '118'].index)
#df = df_subjects[df_subjects['copsacno'] == 102]

def VBA_Cpep(test, display = False):
       t = test["TIMEPOINT"].to_numpy()
       G = test["Glucose"].to_numpy()
       CP = test["C-peptid"].to_numpy()
       # Construt the data struture
       dat = {"t": t.reshape(len(t),1),
              "G": G.reshape(len(t),1),
              "CP": CP.reshape(len(t),1)}

       # Constants
       const = {"dt": 0.1,
              "measCV": 6,
              "age": np.unique(test['age'])[0],
              "subject_type": "normal",
              "CPb": dat["CP"][0, 0],
              "Gb": dat["G"][0, 0]}

       # Construct inversion options
       opt = {"displayWin": display,
              "updateMeasCV": False}

       # Priors [median CV]
       priors = {"T": np.array([10, 50]),                 # min 
              "beta": np.array([20, 50]),              # 1E-9 1/min
              "h": np.array([dat["G"][0,0], 20]),      # mmol/l
              "kd": np.array([1000, 50])}              # 1E-9

       return VBA_OMM.mainCP(dat, priors, const, opt)


out_list = []
R2 = []
RMSE = []
subs = []

subjects = np.unique(df_glu_insulin_Cpep_nonan['copsacno'])

for i in range(len(subjects)):
    test = df_glu_insulin_Cpep[df_glu_insulin_Cpep['copsacno'] == subjects[i]].sort_values('TIMEPOINT')[['copsacno', 'TIMEPOINT', 'age', 'Glucose', 'C-peptid']]
    subs.append(subjects[i])
    out = VBA_Cpep(test)
    out_list.append(out)

    print("\r Process{}%".format(round((i+1)*100/len(subjects))), end="")

os.chdir('/home/zhi/data/CGM/CGM_results/OMM_CP_new')

#with open("out_list", "wb") as fp:   
   # pickle.dump(out_list, fp)

with open("out_list", "rb") as fp:
    out_list = pickle.load(fp)

T_median = [i['posterior']['T'][0] for i in out_list]
beta_median = [i['posterior']['beta'][0] for i in out_list]
kd_median = [i['posterior']['kd'][0] for i in out_list]
h_median = [i['posterior']['h'][0] for i in out_list]

model_results_CP = pd.DataFrame({'copsacno': subjects,
                                'T_median': T_median, 
                        'beta_median': beta_median,
                        'kd_median': kd_median,
           #             'h_median': h_median
                         })

#model_results_CP.to_csv('/home/zhi/data/CGM/CGM_DATA/df_Cpeptid_model_results.csv', index = False)
model_results_CP = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df_Cpeptid_model_results.csv')

df_pheno_all_BMI = df_pheno_all[['copsacno', 'Sex', 'Musclefatratio']].dropna(subset = ['Musclefatratio'])

df_BMI_OMMCP_nonnan = pd.merge(df_pheno_all_BMI, model_results_CP, on = 'copsacno')

par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
rand = 42

scaler2 = StandardScaler()
x_var2 = df_BMI_OMMCP_nonnan.columns[3:]
#X = np.array(df_BMI_OMMG_nonnan[x_var])
scaler2.fit(np.array(df_BMI_OMMCP_nonnan[x_var2]))
X2 = scaler2.transform(np.array(df_BMI_OMMCP_nonnan[x_var2])) 

X_reg2 = np.concatenate((X2, np.array(df_BMI_OMMCP_nonnan[['Sex']])), axis = 1)
#X_reg2 = X2
BMICP_train_SVR, BMICP_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg2, y = preprocessing.scale(df_BMI_OMMCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)
BMICP_train_E, BMICP_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg2, y = preprocessing.scale(df_BMI_OMMCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)

np.mean(BMICP_test_SVR)
np.mean(BMICP_test_E)

model_results_G_CP = pd.merge(model_results, model_results_CP)

df_BMI_OMMGCP_nonnan = pd.merge(df_pheno_all_BMI, model_results_G_CP, on = 'copsacno')


par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
rand = 49

scaler3 = StandardScaler()
x_var3 = df_BMI_OMMGCP_nonnan.columns[3:]
#X = np.array(df_BMI_OMMG_nonnan[x_var])
scaler3.fit(np.array(df_BMI_OMMGCP_nonnan[x_var3]))
X3 = scaler3.transform(np.array(df_BMI_OMMGCP_nonnan[x_var3])) 

X_reg3 = np.concatenate((X3, np.array(df_BMI_OMMGCP_nonnan[['Sex']])), axis = 1)
#X_reg3 = X3
BMIGCP_train_SVR, BMIGCP_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg3, y = preprocessing.scale(df_BMI_OMMGCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)
BMIGCP_train_E, BMIGCP_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg3, y = preprocessing.scale(df_BMI_OMMGCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)

np.mean(BMIGCP_test_SVR)
np.mean([i for i in BMIGCP_test_E if i > 0])
np.mean(BMIglu_test_SVR)
np.mean([i for i in BMIglu_test_E if i > 0])
np.mean(BMICP_test_SVR)
np.mean(BMICP_test_E)

def pipel(random):
       par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
       par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
       rand = random

       scaler = StandardScaler()
       x_var = df_BMI_OMMG_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler.fit(np.array(df_BMI_OMMG_nonnan[x_var]))
       X = scaler.transform(np.array(df_BMI_OMMG_nonnan[x_var])) 

       X_reg = np.concatenate((X, np.array(df_BMI_OMMG_nonnan[['Sex']])), axis = 1)
       #X_reg = X
       BMIglu_train_SVR, BMIglu_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_BMI_OMMG_nonnan['Musclefatratio']), n_beta = False, rand = rand)
       BMIglu_train_E, BMIglu_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_BMI_OMMG_nonnan['Musclefatratio']), n_beta = False, rand = rand)

       scaler1 = StandardScaler()
       x_var1 = df_all.columns[3:]
       scaler1.fit(np.array(df_all[x_var1]))
       X1 = scaler1.transform(np.array(df_all[x_var1])) 

       X_reg1 = np.concatenate((X1, np.array(df_all[['Sex']])), axis = 1)
       #X_reg1 = X1
       BMI_train_SVR, BMI_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg1, y = preprocessing.scale(df_all['Musclefatratio']), n_beta = False, rand = rand)
       BMI_train_E, BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg1, y = preprocessing.scale(df_all['Musclefatratio']), n_beta = False, rand = rand)

       scaler2 = StandardScaler()
       x_var2 = df_BMI_OMMCP_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler2.fit(np.array(df_BMI_OMMCP_nonnan[x_var2]))
       X2 = scaler2.transform(np.array(df_BMI_OMMCP_nonnan[x_var2])) 

       X_reg2 = np.concatenate((X2, np.array(df_BMI_OMMCP_nonnan[['Sex']])), axis = 1)
       #X_reg2 = X2
       BMICP_train_SVR, BMICP_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg2, y = preprocessing.scale(df_BMI_OMMCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)
       BMICP_train_E, BMICP_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg2, y = preprocessing.scale(df_BMI_OMMCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)

       scaler3 = StandardScaler()
       x_var3 = df_BMI_OMMGCP_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler3.fit(np.array(df_BMI_OMMGCP_nonnan[x_var3]))
       X3 = scaler3.transform(np.array(df_BMI_OMMGCP_nonnan[x_var3])) 

       X_reg3 = np.concatenate((X3, np.array(df_BMI_OMMGCP_nonnan[['Sex']])), axis = 1)
       #X_reg3 = X3
       BMIGCP_train_SVR, BMIGCP_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg3, y = preprocessing.scale(df_BMI_OMMGCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)
       BMIGCP_train_E, BMIGCP_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg3, y = preprocessing.scale(df_BMI_OMMGCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)

       print('BMI_test_SVR:', np.mean(BMI_test_SVR))
       print('BMI_test_E:', np.mean([i for i in BMI_test_E if i > 0]))
       print('BMIglu_test_SVR:', np.mean(BMIglu_test_SVR))
       print('BMIglu_test_E:', np.mean([i for i in BMIglu_test_E if i > 0]))
       print('BMICP_test_SVR:', np.mean(BMICP_test_SVR))
       print('BMICP_test_E:', np.mean([i for i in BMICP_test_E if i > 0]))
       print('BMIGCP_test_SVR:', np.mean(BMIGCP_test_SVR))
       print('BMIGCP_test_E:', np.mean([i for i in BMIGCP_test_E if i > 0]))
       return BMI_test_SVR, BMI_test_E, BMIglu_test_SVR, BMIglu_test_E, BMICP_test_SVR, BMICP_test_E, BMIGCP_test_SVR, BMIGCP_test_E

BMI_test_SVR, BMI_test_E, BMIglu_test_SVR, BMIglu_test_E, BMICP_test_SVR, BMICP_test_E, BMIGCP_test_SVR, BMIGCP_test_E = pipel(41)


def pipel2(random):
       par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
       par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
       rand = random

       scaler = StandardScaler()
       x_var = df_BMI_OMMG_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler.fit(np.array(df_BMI_OMMG_nonnan[x_var]))
       X = scaler.transform(np.array(df_BMI_OMMG_nonnan[x_var])) 

       X_reg = np.concatenate((X, np.array(df_BMI_OMMG_nonnan[['Sex']])), axis = 1)
       #X_reg = X
       BMIglu_train_SVR, BMIglu_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_BMI_OMMG_nonnan['Musclefatratio']), n_beta = False, rand = rand)
       #BMIglu_train_E, BMIglu_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_BMI_OMMG_nonnan['Musclefatratio']), n_beta = False, rand = rand)

       scaler1 = StandardScaler()
       x_var1 = df_all.columns[3:]
       scaler1.fit(np.array(df_all[x_var1]))
       X1 = scaler1.transform(np.array(df_all[x_var1])) 

       X_reg1 = np.concatenate((X1, np.array(df_all[['Sex']])), axis = 1)
       #X_reg1 = X1
       BMI_train_SVR, BMI_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg1, y = preprocessing.scale(df_all['Musclefatratio']), n_beta = False, rand = rand)
       #BMI_train_E, BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg1, y = preprocessing.scale(df_all['Musclefatratio']), n_beta = False, rand = rand)

       scaler2 = StandardScaler()
       x_var2 = df_BMI_OMMCP_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler2.fit(np.array(df_BMI_OMMCP_nonnan[x_var2]))
       X2 = scaler2.transform(np.array(df_BMI_OMMCP_nonnan[x_var2])) 

       X_reg2 = np.concatenate((X2, np.array(df_BMI_OMMCP_nonnan[['Sex']])), axis = 1)
       #X_reg2 = X2
       BMICP_train_SVR, BMICP_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg2, y = preprocessing.scale(df_BMI_OMMCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)
       #BMICP_train_E, BMICP_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg2, y = preprocessing.scale(df_BMI_OMMCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)

       scaler3 = StandardScaler()
       x_var3 = df_BMI_OMMGCP_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler3.fit(np.array(df_BMI_OMMGCP_nonnan[x_var3]))
       X3 = scaler3.transform(np.array(df_BMI_OMMGCP_nonnan[x_var3])) 

       X_reg3 = np.concatenate((X3, np.array(df_BMI_OMMGCP_nonnan[['Sex']])), axis = 1)
       #X_reg3 = X3
       BMIGCP_train_SVR, BMIGCP_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg3, y = preprocessing.scale(df_BMI_OMMGCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)
       #BMIGCP_train_E, BMIGCP_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg3, y = preprocessing.scale(df_BMI_OMMGCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)

       return BMI_train_SVR, BMI_test_SVR, BMIglu_train_SVR, BMIglu_test_SVR, BMICP_train_SVR, BMICP_test_SVR, BMIGCP_train_SVR, BMIGCP_test_SVR

BMI_train_SVR, BMI_test_SVR, BMIglu_train_SVR, BMIglu_test_SVR, BMICP_train_SVR, BMICP_test_SVR, BMIGCP_train_SVR, BMIGCP_test_SVR = pipel2(41)

def pipel3(random, pheno):
       par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
       par_gridsvr = {'C': [0.01, 0.1, 1, 10, 100]}
       rand = random

       df_pheno_all_BMI = df_pheno_all[['copsacno', 'Sex', pheno]].dropna(subset = [pheno])

       df_BMI_OMMG_nonnan = pd.merge(df_pheno_all_BMI, model_results, on = 'copsacno')
       scaler = StandardScaler()
       x_var = df_BMI_OMMG_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler.fit(np.array(df_BMI_OMMG_nonnan[x_var]))
       X = scaler.transform(np.array(df_BMI_OMMG_nonnan[x_var])) 

       X_reg = np.concatenate((X, np.array(df_BMI_OMMG_nonnan[['Sex']])), axis = 1)
       #X_reg = X
       BMIglu_train_SVR, BMIglu_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_BMI_OMMG_nonnan[pheno]), n_beta = False, rand = rand)
       #BMIglu_train_E, BMIglu_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_BMI_OMMG_nonnan[pheno]), n_beta = False, rand = rand)

       df_BMI_gluT_nonnan = pd.merge(df_pheno_all_BMI, df_glu_insulin_nonan_pivot, on = 'copsacno')
       df_BMI_gluT_nonnan.columns = ['{}_{}'.format(x[0], x[1]) for x in df_BMI_gluT_nonnan.columns]
       df_BMI_gluT_nonnan.rename(columns = {df_BMI_gluT_nonnan.columns[0]: 'copsacno', df_BMI_gluT_nonnan.columns[1]: 'Sex', df_BMI_gluT_nonnan.columns[2]: pheno},inplace=True)
       df_BMI_gluT_nonnan.reset_index(drop = True, inplace = True)

       df_BMI_diet_nonnan = pd.merge(df_pheno_all_BMI, df_diet_noheader_nonan_pivot, on = 'copsacno')
       df_BMI_diet_nonnan.columns = ['{}_{}'.format(x[0], x[1]) for x in df_BMI_diet_nonnan.columns]
       df_BMI_diet_nonnan.rename(columns = {df_BMI_diet_nonnan.columns[0]: 'copsacno', df_BMI_diet_nonnan.columns[1]: 'Sex', df_BMI_diet_nonnan.columns[2]: pheno},inplace=True)
       df_BMI_diet_nonnan.reset_index(drop = True, inplace = True)

       df_all = pd.merge(df_BMI_gluT_nonnan, df_BMI_diet_nonnan, on = ['copsacno', 'Sex', pheno])
       
       scaler1 = StandardScaler()
       x_var1 = df_all.columns[3:]
       scaler1.fit(np.array(df_all[x_var1]))
       X1 = scaler1.transform(np.array(df_all[x_var1])) 

       X_reg1 = np.concatenate((X1, np.array(df_all[['Sex']])), axis = 1)
       #X_reg1 = X1
       BMI_train_SVR, BMI_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg1, y = preprocessing.scale(df_all[pheno]), n_beta = False, rand = rand)
      # BMI_train_E, BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg1, y = preprocessing.scale(df_all['Musclefatratio']), n_beta = False, rand = rand)

       df_BMI_OMMCP_nonnan = pd.merge(df_pheno_all_BMI, model_results_CP, on = 'copsacno')
       scaler2 = StandardScaler()
       x_var2 = df_BMI_OMMCP_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler2.fit(np.array(df_BMI_OMMCP_nonnan[x_var2]))
       X2 = scaler2.transform(np.array(df_BMI_OMMCP_nonnan[x_var2])) 

       X_reg2 = np.concatenate((X2, np.array(df_BMI_OMMCP_nonnan[['Sex']])), axis = 1)
       #X_reg2 = X2
       BMICP_train_SVR, BMICP_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg2, y = preprocessing.scale(df_BMI_OMMCP_nonnan[pheno]), n_beta = False, rand = rand)
       #BMICP_train_E, BMICP_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg2, y = preprocessing.scale(df_BMI_OMMCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)

       model_results_G_CP = pd.merge(model_results, model_results_CP)
       df_BMI_OMMGCP_nonnan = pd.merge(df_pheno_all_BMI, model_results_G_CP, on = 'copsacno')
       scaler3 = StandardScaler()
       x_var3 = df_BMI_OMMGCP_nonnan.columns[3:]
       #X = np.array(df_BMI_OMMG_nonnan[x_var])
       scaler3.fit(np.array(df_BMI_OMMGCP_nonnan[x_var3]))
       X3 = scaler3.transform(np.array(df_BMI_OMMGCP_nonnan[x_var3])) 

       X_reg3 = np.concatenate((X3, np.array(df_BMI_OMMGCP_nonnan[['Sex']])), axis = 1)
       #X_reg3 = X3
       BMIGCP_train_SVR, BMIGCP_test_SVR = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg3, y = preprocessing.scale(df_BMI_OMMGCP_nonnan[pheno]), n_beta = False, rand = rand)
       #BMIGCP_train_E, BMIGCP_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg3, y = preprocessing.scale(df_BMI_OMMGCP_nonnan['Musclefatratio']), n_beta = False, rand = rand)

       print('BMI_test_SVR:', np.mean(BMI_test_SVR))
      # print('BMI_test_E:', np.mean([i for i in BMI_test_E if i > 0]))
       print('BMIglu_test_SVR:', np.mean(BMIglu_test_SVR))
      # print('BMIglu_test_E:', np.mean([i for i in BMIglu_test_E if i > 0]))
       print('BMICP_test_SVR:', np.mean(BMICP_test_SVR))
       #print('BMICP_test_E:', np.mean([i for i in BMICP_test_E if i > 0]))
       print('BMIGCP_test_SVR:', np.mean(BMIGCP_test_SVR))
       #print('BMIGCP_test_E:', np.mean([i for i in BMIGCP_test_E if i > 0]))
       return BMI_train_SVR, BMI_test_SVR, BMIglu_train_SVR, BMIglu_test_SVR, BMICP_train_SVR, BMICP_test_SVR, BMIGCP_train_SVR, BMIGCP_test_SVR

BMI_train_SVR, BMI_test_SVR, BMIglu_train_SVR, BMIglu_test_SVR, BMICP_train_SVR, BMICP_test_SVR, BMIGCP_train_SVR, BMIGCP_test_SVR = pipel3(41, 'Musclefatratio')

BMI_train_SVR, BMI_test_SVR, BMIglu_train_SVR, BMIglu_test_SVR, BMICP_train_SVR, BMICP_test_SVR, BMIGCP_train_SVR, BMIGCP_test_SVR = pipel3(36, 'Fat_percent')

BMI_train_SVR, BMI_test_SVR, BMIglu_train_SVR, BMIglu_test_SVR, BMICP_train_SVR, BMICP_test_SVR, BMIGCP_train_SVR, BMIGCP_test_SVR = pipel3(45, 'Muscle_mass')

BMI_train_SVR, BMI_test_SVR, BMIglu_train_SVR, BMIglu_test_SVR, BMICP_train_SVR, BMICP_test_SVR, BMIGCP_train_SVR, BMIGCP_test_SVR = pipel3(41, 'Skelmuscle_mass')

BMI_train_SVR, BMI_test_SVR, BMIglu_train_SVR, BMIglu_test_SVR, BMICP_train_SVR, BMICP_test_SVR, BMIGCP_train_SVR, BMIGCP_test_SVR = pipel3(47, 'Bone_mass')

labels = ['Metabolites', 'Gluocse Insulin model', 'Gluocse C-peptid model', 'Combine models']
dpi = 1000
title = 'Support Vector Regression results of Bone_mass'

x = np.arange(len(labels))
train_mean = [round(np.mean(BMI_train_SVR), 3), round(np.mean(BMIglu_train_SVR), 3), round(np.mean(BMICP_train_SVR), 3), round(np.mean(BMIGCP_train_SVR), 3)]
test_mean = [round(np.mean(BMI_test_SVR), 3), round(np.mean(BMIglu_test_SVR), 3), round(np.mean(BMICP_test_SVR), 3), round(np.mean(BMIGCP_test_SVR), 3)]
train_std = [np.std(BMI_train_SVR),  np.std(BMIglu_train_SVR), np.std(BMICP_train_SVR),  np.std(BMIGCP_train_SVR)]
test_std = [np.std(BMI_test_SVR), np.std(BMIglu_test_SVR), np.std(BMICP_test_SVR), np.std(BMIGCP_test_SVR)]    

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
ax.set_xticklabels(labels, fontsize = 6)
ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
#for i, v in enumerate(train_mean):
   # rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
#for i, v in enumerate(test_mean):
    #rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
fig.tight_layout()

plt.show()   

R2 = [i['Performance']['R2'] for i in out_list]
RMSE = [i['Performance']['RMSE'] for i in out_list]

fig, ax = plt.subplots(dpi = 1000)
ax.set_title('Distribution of R2 for each subject')
plt.hist(R2, bins = 100)

sns.set(rc={"figure.dpi":1000})
sns.set_style("ticks")
sns.lineplot(data=df_glu_insulin_Cpep_nonan, x='TIMEPOINT', y='C-peptid')
#sns.lineplot(data=df_glu_insulin, x='TIMEPOINT', y='Glucose')
#sns.lineplot(data=df_glu_insulin, x='TIMEPOINT', y='Insulin')
plt.legend(loc='upper right', labels=['Mean ± Std'])


import numpy as np
import sys
sys.path.append('/home/zhi/data/CGM/CGM_Project/VBA_OMM_Python/')
import VBA_OMM
sys.path.append('/home/zhi/data/CGM/CGM_Project/VBA_OMM_Python/VBA_OMM/')
import VBA_OMM_G
import csv
import pickle
import os
import seaborn as sns

# Read the demo data from csv file
with open('demo_dat.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    t = np.zeros((1, 19))
    G = np.zeros((1, 19))
    I = np.zeros((1, 19))
    i = 0
    for row in csv_reader:
        t[0, i] = row[0]
        G[0, i] = row[1]
        I[0, i] = row[2]
        i=i+1

def VBA_glu(test, method = 'RaPL', display = False):
    t = np.zeros((1, 8))
    G = np.zeros((1, 8))
    I = np.zeros((1, 8))
    i = 0
    for index, row in test.iterrows():
        t[0, i] = row['TIMEPOINT']
        G[0, i] = row['Glucose']
        I[0, i] = row['Insulin']
        i=i+1

    # Construt the data struture
    dat = {"t": t,
        "G": G,
        "I": I}

    # Constants
    const = {"A": 6,
            "V": 0.145,
            "dt": 0.1,
            "Rap": [],
            "X0": 0,
            "measCV": 2,
            "Gb": dat["G"][0, 0],
            "G0": dat["G"][0, 0],
            "Ib": dat["I"][0, 0]}

    # Construct inversion options
    opt = {"GA_fun": method,
        "tb": np.array([0, 15, 30, 60, 90, 120, 150, 240]),
        #"tb": np.array([0, 10, 30, 60, 90, 120, 180, 300]),
        "alpha": 0.017,
        "displayWin": display}

    # Priors
        # - System Parameters [median CV]
    priors = {"p1": np.array([0.025, 25]),
            "p2": np.array([0.012, 40]),
            "SI": np.array([7.1E-4, 100])}

        # - Input function Parameters
    if opt["GA_fun"] == 'RaPL':
        priors.update({"k": np.array([[3.2E-3*const["A"], 50],
                                    [7.3E-3*const["A"], 50],
                                    [5.4E-3*const["A"], 50],
                                    [5.1E-3*const["A"], 50],
                                    [3.7E-3*const["A"], 50],
                                    [1.8E-3*const["A"], 50]])})
    if opt["GA_fun"] == 'RaLN':
        priors.update({"k": np.array([[30, 30],
                                    [0.5, 30],
                                    [100, 30],
                                    [0.5, 30],
                                    [0.7, 30]])})

    return VBA_OMM_G.main(dat, priors, const, opt)
    
out = VBA_glu(test1)

test2 = df_glu_insulin[df_glu_insulin['copsacno'] == '303'].sort_values('TIMEPOINT')[['copsacno', 'TIMEPOINT', 'Glucose', 'Insulin']]
VBA_glu(test2, display = True)

subjects = np.unique(df_glu_insulin['copsacno'])
df_glu_insulin['Insulin'] = df_glu_insulin['Insulin2']

p1_mean = []
p2_mean = []
SI_mean = []

p1_CV = []
p2_CV = []
SI_CV = []

out_list = []
subs = []
for i in range(len(subjects)):
    test = df_glu_insulin[df_glu_insulin['copsacno'] == subjects[i]].sort_values('TIMEPOINT')[['copsacno', 'TIMEPOINT', 'Glucose', 'Insulin']]
    if (test.shape[0] == 8):
        subs.append(subjects[i])
        #out = VBA_glu(test, method = 'RaLN')
        #out_list.append(out)
        #p1_mean.append(out['posterior']['p1'][0])
        #p2_mean.append(out['posterior']['p2'][0])
        #SI_mean.append(out['posterior']['SI'][0])
        #p1_CV.append(out['posterior']['p1'][1])
        #p2_CV.append(out['posterior']['p2'][1])
        #SI_CV.append(out['posterior']['SI'][1])

    print("\r Process{}%".format(round((i+1)*100/len(subjects))), end="")


os.chdir('/home/zhi/data/CGM/CGM_results')
with open("out_list2", "wb") as fp:   
    pickle.dump(out_list, fp)

with open("out_list1", "rb") as fp:
    out_list_PL = pickle.load(fp)

with open("out_list2", "rb") as fp:
    out_list_LN = pickle.load(fp)

#with open("p1_mean", "wb") as fp:   
   # pickle.dump(p1_mean, fp)
with open("p1_mean", "rb") as fp:
    p1_median = pickle.load(fp)

#with open("p2_mean", "wb") as fp:   
 #   pickle.dump(p2_mean, fp)
with open("p2_mean", "rb") as fp:
    p2_median = pickle.load(fp)

#with open("SI_mean", "wb") as fp:   
    #pickle.dump(SI_mean, fp)
with open("SI_mean", "rb") as fp:
    SI_median = pickle.load(fp)

#with open("p1_CV", "wb") as fp:   
   # pickle.dump(p1_CV, fp)
with open("p1_CV", "rb") as fp:
    p1_CV = pickle.load(fp)

#with open("p2_CV", "wb") as fp:   
   # pickle.dump(p2_CV, fp)
with open("p2_CV", "rb") as fp:
    p2_CV = pickle.load(fp)

#with open("SI_CV", "wb") as fp:   
  #  pickle.dump(SI_CV, fp)
with open("SI_CV", "rb") as fp:
    SI_CV = pickle.load(fp)

k1 = []
k2 = []
k3 = []
k4 = []
k5 = []
k6 = []
for i in range(len(out_list_PL)):
    k1.append()


df_glu_insulin['Insulin'] = df_glu_insulin['Insulin2']

sns.set(rc={"figure.dpi":1000})
sns.set_style("ticks")
sns.lineplot(data=df_glu_insulin, x='TIMEPOINT', y='C-pipetid')
#sns.lineplot(data=df_glu_insulin, x='TIMEPOINT', y='Glucose')
#sns.lineplot(data=df_glu_insulin, x='TIMEPOINT', y='Insulin')
plt.legend(loc='upper right', labels=['Mean ± Std'])


p1_median = [i['posterior']['p1'][0] for i in out_list_PL]
p2_median = [i['posterior']['p2'][0] for i in out_list_PL]
SI_median = [i['posterior']['SI'][0] for i in out_list_PL]

p1_CV = [i['posterior']['p1'][1] for i in out_list_PL]
p2_CV = [i['posterior']['p2'][1] for i in out_list_PL]
SI_CV = [i['posterior']['SI'][1] for i in out_list_PL]

R2 = [i['Performance']['R2'] for i in out_list_PL]
RMSE = [i['Performance']['RMSE'] for i in out_list_PL]

results = pd.DataFrame({'p1 median': p1_median, 
                        'p2 median': p2_median,
                        'SI median': SI_median})

results = pd.DataFrame({'p1 CV': p1_CV, 
                        'p2 CV': p2_CV,
                        'SI CV': SI_CV})

results = pd.DataFrame({'R2': R2})
results = pd.DataFrame({'RMSE': RMSE})

plt.hist(results)

ax = sns.boxplot(data=results)
 
# Add jitter with the swarmplot function
ax = sns.swarmplot(data=results)
plt.show()

np.corrcoef(p1_median, p2_median)
np.corrcoef(p1_median, SI_median)
np.corrcoef(p2_median, SI_median)

np.corrcoef(p1_CV, p2_CV)
np.corrcoef(p1_CV, SI_CV)
np.corrcoef(p2_CV, SI_CV)

corr = results.corr()
corr.style.background_gradient(cmap='coolwarm')

model_results = pd.DataFrame({'copsacno': subs,
                                'p1 median': p1_median, 
                        'p2 median': p2_median,
                        'SI median': SI_median,
                        'p1 CV': p1_CV, 
                        'p2 CV': p2_CV,
                        'SI CV': SI_CV,
                        #'R2': R2,
                        #'RMSE': RMSE
                        })
                    
df_pheno = pd.read_csv('~/data/CGM/CGM_DATA/CGM3.csv')
df_pheno['copsacno'] = df_pheno['copsacno'].astype(str)
df_pheno = df_pheno.drop_duplicates(subset="copsacno")
df_model_pheno_model = pd.merge(model_results, df_pheno, on = 'copsacno')
df_model_pheno_model = pd.get_dummies(df_model_pheno_model, columns=['Race'], drop_first = True)
df_model_pheno_model = df_model_pheno_model.fillna(method = 'backfill')

df_model_pheno_new = df_model_pheno[['p1 CV', 'p2 CV', 'SI CV', 'Sex_binary']]
df_model_pheno_m = df_model_pheno[df_model_pheno['Sex_binary'] == 1][['p1 median', 'p2 median', 'SI median', 'Sex_binary']]
df_model_pheno_f = df_model_pheno[df_model_pheno['Sex_binary'] == 0][['p1 median', 'p2 median', 'SI median', 'Sex_binary']]

df_long = pd.melt(df_model_pheno_new, "Sex_binary", var_name="a", value_name="c")
g = sns.factorplot("a", hue="Sex_binary", y="c", data=df_long, kind="box")
legend = g._legend

legend.set_title("Sex")
for t, l in zip(legend.texts,("Male", "Female")):
    t.set_text(l)

ax = sns.boxplot(data=df_model_pheno_m, x = df_model_pheno_m.columns[:3], hue = "Sex_binary")
 
# Add jitter with the swarmplot function
ax = sns.swarmplot(data=df_model_pheno_m['p1 median', 'p2 median', 'SI median'], hue = "Sex_binary")
plt.show()

scaler = StandardScaler()
#x_var = ['T1', 'T2', 'T3', 'SDRC', 'GMI', 'JIndex']
x_var = ['p1 median', 'p2 median', 'SI median', 'p1 CV', 'p2 CV',
       'SI CV', 'householdincome', 'm_edu', 'Mother_age_2yrs']
x_var = ['p1 median', 'p2 median', 'SI median', 'p1 CV', 'p2 CV',
       'SI CV']
scaler.fit(np.array(df_model_pheno[x_var]))
X = scaler.transform(np.array(df_model_pheno[x_var])) 
X = np.array(df_model_pheno[x_var])
X_reg = np.concatenate((X, np.array(df_model_pheno[['Sex_binary']])), axis = 1)
#X_reg = np.concatenate((np.array(df_model_pheno[['Sex_binary']]), np.array(df_model_pheno[['Race_non-caucasian']])), axis = 1)
X_reg = X
#BMI_train, BMI_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_new_outcome['bmi18y']), n_beta = False, rand = 66)
BMI_train, BMI_test, beta_BMI = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno['bmi18y']), n_beta = 11, rand = 76)
MFratio_train, MFratio_test, beta_MFratio = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno['Musclefatratio']), n_beta = 8, rand = 99)
Fit_train, Fit_test, beta_Fit = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno['FITNESS']), n_beta = 11, rand = 66)
MuscleMass_train, MuscleMass_test, beta_MuscleMass = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno['Muscle_mass']), n_beta = 11, rand = 99)
WHR_train, WHR_test, beta_WHR = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno['waisthipratio']), n_beta = 11, rand = 99)
Bonemass_train, Bonemass_test, beta_Bonemass = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno['Bone_mass']), n_beta = 11, rand = 99)
Tfat_train, Tfat_test, beta_Tfat = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno['Trunkfat_percent']), n_beta = 11, rand = 99)
Sk_train, Sk_test, Sk_Vfat = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = X_reg, y = preprocessing.scale(df_model_pheno['Skelmuscle_mass']), n_beta = 11, rand = 99)


#MuscleMass_train, MuscleMass_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_new_outcome['Muscle_mass']), n_beta = False, rand = 66)

BMI_train, BMI_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_model_pheno['bmi18y']), n_beta = False, rand = 66)
MFratio_train, MFratio_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_model_pheno['Musclefatratio']), n_beta = False, rand = 99)
Fit_train, Fit_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_model_pheno['FITNESS']), n_beta = False, rand = 66)
MuscleMass_train, MuscleMass_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_model_pheno['Muscle_mass']), n_beta = False, rand = 99)
WHR_train, WHR_test = CV(p_grid = par_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = X_reg, y = preprocessing.scale(df_model_pheno['waisthipratio']), n_beta = False, rand = 99)

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

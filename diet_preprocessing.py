import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_diet = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/NAS_data/Diet_chanllenge_BiochemData.csv')
df_diet = df_diet.rename(columns = {"COPSACNO":"copsacno"})
df_diet_noheader = df_diet.tail(df_diet.shape[0] - 4)
df_diet_noheader['copsacno'] = df_diet_noheader['copsacno'].astype(str)
df_diet_noheader.reset_index(drop = True, inplace = True)

df_diet_noheader = df_diet_noheader.drop(df_diet_noheader.columns[0:3], axis = 1)
df_diet_noheader = df_diet_noheader.drop(df_diet_noheader.columns[[2,3]], axis = 1)
df_diet_noheader_nonan = df_diet_noheader.drop(['Phe', 'XXL-VLDL-PL %', 'XXL-VLDL-C %', 'XXL-VLDL-CE %', 'XXL-VLDL-FC %', 'XXL-VLDL-TG %'], axis = 1)
df_diet_noheader_nonan = df_diet_noheader_nonan.dropna()
df_diet_noheader_nonan = df_diet_noheader_nonan.groupby('copsacno').filter(lambda x: len(x) == 8)
df_BMI_diet_nonnan = pd.merge(df_pheno_BMI, df_diet_noheader_nonan, on = 'copsacno')
df_BMI_diet_nonnan.reset_index(drop = True, inplace = True)




df_diet_group = df_diet_noheader_nonan.drop_duplicates(subset = 'copsacno')
df_diet_group.reset_index(drop = True, inplace = True)

na = df_diet_noheader.isna().sum() 

df_diet_group = df_diet_noheader.drop_duplicates(subset = 'copsacno')
df_diet_group.reset_index(drop = True, inplace = True)
df_diet_group_metabolism = df_diet_group[]

df_pheno = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/CGM3.csv')

df_pheno1 = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/COPSAC2000_DietChallenge_MetaData.csv')
df_pheno1['copsacno'] = df_pheno1['pid2'].astype(str)
df_pheno1[df_pheno1['copsacno'] == '4']

df_pheno['copsacno'] = df_pheno['copsacno'].astype(str)
df_pheno_group = df_pheno.drop_duplicates(subset = 'copsacno')
df_pheno_group_noNAN = df_pheno_group.dropna(subset = ['bmi18y', 'Sex_binary'])
df_pheno_group_noNAN.reset_index(drop = True, inplace = True)

df_diet_pheno1 = pd.merge(df_pheno1, df_pheno_group_noNAN, on = 'copsacno')
df_diet_pheno = pd.merge(df_diet_group, df_pheno_group_noNAN, on = 'copsacno')

df_pheno = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/COPSAC2000_pheno.csv')
df_pheno['copsacno'] = df_pheno['COPSACNO'].astype(str)
df_pheno_BMI = df_pheno[['copsacno', 'bmi18y']].dropna(subset = ['bmi18y'])
df_pheno_BMI.reset_index(drop = True, inplace = True)

df_BMI_diet = pd.merge(df_pheno_BMI, df_diet_group, on = 'copsacno')

df_pheno_WHR = df_pheno[['copsacno', 'WAIST', 'HIPCIRCUM']].dropna(subset = ['WAIST', 'HIPCIRCUM'])
df_pheno_WHR['waist-hip-ratio'] = df_pheno_WHR['WAIST']/df_pheno_WHR['HIPCIRCUM']
df_WHR_diet_nonnan = pd.merge(df_pheno_WHR, df_diet_noheader_nonan, on = 'copsacno')
df_WHR_diet_nonnan.reset_index(drop = True, inplace = True)

df_pheno_all = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df2000.csv')
df_pheno_all = df_pheno_all.rename(columns = {"COPSACNO":"copsacno"})
df_pheno_all['copsacno'] = df_pheno_all['copsacno'].astype(str)
df_pheno_all_BMI = df_pheno_all[['copsacno', 'BMI']].dropna(subset = ['BMI'])
df_BMI_diet1 = pd.merge(df_pheno_all_BMI, df_diet_group, on = 'copsacno')


df_pheno_all_Musclefatratio = df_pheno_all[['copsacno', 'Musclefatratio']].dropna(subset = ['Musclefatratio'])
df_Musclefatratio_diet = pd.merge(df_pheno_all_Musclefatratio, df_diet_group, on = 'copsacno')

a = 'adhd_score'
df_pheno_all = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df2000.csv')
df_pheno_all = df_pheno_all.rename(columns = {"COPSACNO":"copsacno"})
df_pheno_all['copsacno'] = df_pheno_all['copsacno'].astype(str)
df_pheno_all_Musclefatratio = df_pheno_all[['copsacno', a]].dropna(subset = [a])
df_Musclefatratio_diet = pd.merge(df_pheno_all_Musclefatratio, df_diet_group, on = 'copsacno')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu

df_PRS = pd.read_excel('~/data/CGM/CGM_DATA/pgs_all_metadata.xlsx', sheet_name = 3)
df_PRS_asthma = pd.read_csv('~/data/CGM/CGM_DATA/PGS001849.sscore', header = None, sep = '\t', names = ["FID","IID","x1","x2","x3","SCORE"])
df_PRS_asthma = df_PRS_asthma.tail(-1)

df_PRS_asthma_endpoint = pd.read_csv('~/data/CGM/CGM_DATA/PRS_asthma.csv')
df_PRS_asthma_endpoint_2000 = df_PRS_asthma_endpoint[df_PRS_asthma_endpoint['BIRTHCOHORT'] == 1]
#df_PRS_asthma_endpoint_2010 = df_PRS_asthma_endpoint[df_PRS_asthma_endpoint['BIRTHCOHORT'] == 0]
df_PRS_asthma_endpoint_2000 = df_PRS_asthma_endpoint_2000.rename(columns = {'ID': 'copsacno'})
df_glu_insulin_8 = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df_biochem_insulin_Cpeptid.csv')

df_PRS_asthma_endpoint_biochem = pd.merge(df_PRS_asthma_endpoint_2000[['copsacno', 'SCORE', 'event_j45_3yr']], df_glu_insulin_8, on = 'copsacno')
df_PRS_asthma_endpoint_biochem_norep = df_PRS_asthma_endpoint_biochem.drop_duplicates(subset=['copsacno'])

np.unique(df_PRS_asthma_endpoint_biochem['event_j45_3yr'], return_counts = True)
plt.hist(df_PRS_asthma_endpoint_2000['SCORE'])
plt.scatter(df_PRS_asthma_endpoint_biochem_norep['event_j45_3yr'], df_PRS_asthma_endpoint_biochem_norep['SCORE'], alpha=0.25)

#find rows contain string 'asthma' in column 'Mapped Trait(s) (EFO label)' from df_PRS
df_PRS_asthma = df_PRS[df_PRS['Mapped Trait(s) (EFO label)'].str.contains('asthma')]

a = np.array([  1,   4,   5,   8,  10,  11,  23,  25,  26,  33,  36,  38,  39,
         43,  44,  48,  49,  50,  52,  53,  55,  56,  58,  63,  64,  66,
         68,  70,  74,  75,  76,  77,  78,  81,  87,  88,  90,  91,  94,
         96,  97, 103, 104, 105, 106, 108, 110, 112, 117, 119, 121, 125,
        127, 130, 131, 132, 133, 136, 137, 138, 139, 142, 145, 147, 153,
        154, 161, 164, 167, 168, 169, 172, 175, 176, 181, 184, 189, 190,
        191, 193, 194, 195, 197, 199, 200, 202, 203, 205, 206, 207, 208,
        210, 213, 216, 218, 224, 230, 232, 237, 243, 244, 245, 247, 249,
        252, 259, 261, 262, 263, 265, 267, 269, 274, 276, 277, 279, 280,
        281, 282, 284, 290, 291, 295, 297, 298, 302, 312, 316, 321, 322,
        327, 328, 329, 330, 334, 339, 341, 364, 365, 368, 369, 376, 378,
        379, 381, 383, 388, 389, 394, 396, 401, 406, 409, 413, 415, 417,
        420, 424, 427, 430, 431, 432, 433, 434, 435, 438, 440, 442, 444,
        445, 447, 449, 450])

df_a = pd.DataFrame(a, columns = ['copsacno'])

df_PRS_asthma_endpoint_biochem = pd.merge(df_a, df_glu_insulin_8, on = 'copsacno')
df_PRS_asthma_endpoint_biochem_norep = df_PRS_asthma_endpoint_biochem.drop_duplicates(subset=['copsacno'])

df_glu_insulin_8 = pd.read_csv('/home/zhi/nas/Diet_challenge/df_biochem_insulin_Cpeptid_filtered.csv')
df_asthma18 = pd.read_excel('/home/zhi/nas/Diet_challenge/CP0155_Asthma_cross_18yr.xlsx', sheet_name = 'Data')
df_asthma19 = pd.read_excel('/home/zhi/nas/Diet_challenge/CP0156_Asthma_diagnoses_0-19yr.xlsx', sheet_name = 'Data')

df_asthma19['copsacno'] = df_asthma19['copsacno'].astype('str')
df_asthma19['icd10'] = df_asthma19['icd10'].astype('str')
df_asthma19['copsacno_icd10'] = df_asthma19['copsacno'] + '_' + df_asthma19['icd10']
df_glu_insulin_8['copsacno'] = df_glu_insulin_8['copsacno'].astype('str')

df_asthma = df_glu_insulin_8[['copsacno']].drop_duplicates(subset = ['copsacno'], keep = 'first')
df_asthma.reset_index(drop = True, inplace = True)
df_asthma['asthma'] = 0
df_asthma['copsacno'] = df_asthma['copsacno'].astype('str')

#mark df_asthma as 1 if its copsacno is in df_asthma19
for i in range(len(df_asthma)):
    if df_asthma['copsacno'][i] in df_asthma19['copsacno'].values:
        df_asthma['asthma'][i] = 1
#find is there any missing value in df_glucose_insulin_8 and find where are them
na = pd.DataFrame(df_glu_insulin_8.isnull().sum())
na[na[0] != 0]

df_2000 = pd.read_csv('/home/zhi/nas/Diet_challenge/df2000.csv')
df_2000['copsacno'] = df_2000['COPSACNO'].astype('str')
df_asthma_sex = pd.merge(df_asthma, df_2000[['Sex', 'copsacno']], on = 'copsacno', how = 'left')

#find the row and column with nan from df_glu_insulin_8
df_glu_insulin_8[df_glu_insulin_8.isnull().any(axis = 1)][['copsacno', 'TIMEPOINT', 'Gly', 'Tyr', 'Lactate', 'Pyruvate', 'bOHbutyrate', 'Creatinine']]
#drop rows with more than 2 nan
df_glu_insulin_8.drop(df_glu_insulin_8[df_glu_insulin_8['copsacno'] == '182'].index, inplace = True)
df_glu_insulin_8.drop(df_glu_insulin_8[df_glu_insulin_8['copsacno'] == '316'].index, inplace = True)
df_glu_insulin_8.reset_index(drop = True, inplace = True)

#drop TAG
na1 = pd.DataFrame(df_glu_insulin_8.applymap(lambda x: 'TAG' in str(x)).sum())
na1[na1[0] != 0]

df_glu_insulin_8[df_glu_insulin_8.applymap(lambda x: 'TAG' in str(x)).any(axis = 1)][['copsacno', 'TIMEPOINT'] + na1[na1[0] != 0].index.tolist()]
df_glu_insulin_8.drop(df_glu_insulin_8[df_glu_insulin_8['copsacno'] == '142'].index, inplace = True)
df_glu_insulin_8.drop(df_glu_insulin_8[df_glu_insulin_8['copsacno'] == '79'].index, inplace = True)
df_glu_insulin_8.reset_index(drop = True, inplace = True)

df_biochem_asthma = pd.merge(df_asthma_sex, df_glu_insulin_8, on = 'copsacno', how = 'right')
df_biochem_asthma['copsacno'] = df_biochem_asthma['copsacno'].astype('str')
df_biochem_asthma['asthma'] = df_biochem_asthma['asthma'].astype('str')
len(np.unique(df_biochem_asthma['copsacno']))

df_phenotype = df_biochem_asthma.copy(deep = True).drop_duplicates(subset = ['copsacno'], keep = 'first')[['copsacno', 'asthma', 'Sex', 'age']]
np.unique(df_phenotype['asthma'], return_counts = True)
np.unique(df_phenotype['Sex'], return_counts = True)
np.unique(df_phenotype[df_phenotype['Sex'] == 0]['asthma'], return_counts = True)
np.unique(df_phenotype[df_phenotype['Sex'] == 1]['asthma'], return_counts = True)

#pivot df_biochem_asthma after column age to get the dataframe with only one row for each copsacno
df_biochem = df_biochem_asthma.copy(deep = True)
df_biochem.drop(columns = ['asthma', 'age', 'Sex'], inplace = True)
df_biochem_pivot = df_biochem.pivot(index='copsacno', columns='TIMEPOINT')
df_biochem_pivot.columns = ['{}_{}'.format(x[0], x[1]) for x in df_biochem_pivot.columns]
df_biochem_pivot = pd.merge(df_phenotype, df_biochem_pivot, on = 'copsacno')
df_biochem_pivot.reset_index(drop = True, inplace = True)

row, cols = np.where(df_biochem_pivot.applymap(lambda x: 'TAG' in str(x)))
row1, cols1 = np.where(pd.isnull(df_biochem_pivot))

def impute_nan(r, c, df):
    for id in r:
        gender = df.loc[id]['Sex']
        for ix in c:
            df1 = df[df['Sex'] == gender].iloc[:,ix]
            df.iloc[id, ix] = np.nanmedian(df1[df1 != 'TAG'].astype('float'))
    return df

df_biochem_pivot_noTAG = impute_nan(row, cols, df_biochem_pivot)
df_biochem_pivot_noTAG_nonan = impute_nan(row1, cols1, df_biochem_pivot)

df_biochem_pivot_noTAG_nonan.to_csv('/home/zhi/nas/Diet_challenge/df_biochem_pheno_imputed.csv')
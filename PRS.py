import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df_PRS = pd.read_excel('~/data/CGM/CGM_DATA/pgs_all_metadata.xlsx', sheet_name = 4)
df_PRS_asthma = pd.read_csv('~/data/CGM/CGM_DATA/PGS001849.sscore', header = None, sep = '\t', names = ["FID","IID","x1","x2","x3","SCORE"])
df_PRS_asthma = df_PRS_asthma.tail(-1)

df_PRS_asthma_endpoint = pd.read_csv('~/data/CGM/CGM_DATA/PRS_asthma.csv')
df_PRS_asthma_endpoint_2000 = df_PRS_asthma_endpoint[df_PRS_asthma_endpoint['BIRTHCOHORT'] == 1]
df_PRS_asthma_endpoint_2010 = df_PRS_asthma_endpoint[df_PRS_asthma_endpoint['BIRTHCOHORT'] == 0]
df_PRS_asthma_endpoint_2000 = df_PRS_asthma_endpoint_2000.rename(columns = {'ID': 'copsacno'})
df_glu_insulin_8 = pd.read_csv('/home/zhi/data/CGM/CGM_DATA/df_biochem_insulin_Cpeptid.csv')

df_PRS_asthma_endpoint_biochem = pd.merge(df_PRS_asthma_endpoint_2000[['copsacno', 'SCORE', 'event_j45_3yr']], df_glu_insulin_8, on = 'copsacno')

np.unique(df_PRS_asthma_endpoint_biochem['event_j45_3yr'], return_counts = True)
plt.hist(df_PRS_asthma_endpoint_2000['SCORE'])

plt.scatter(df_PRS_asthma_endpoint_2010['event_j45_3yr'], df_PRS_asthma_endpoint_2010['SCORE'], alpha=0.05)

stats.ttest_ind(df_PRS_asthma_endpoint_2000[df_PRS_asthma_endpoint_2000['event_j45_3yr'] == 0]['SCORE'], df_PRS_asthma_endpoint_2000[df_PRS_asthma_endpoint_2000['event_j45_3yr'] == 1]['SCORE'])
stats.ttest_ind(df_PRS_asthma_endpoint_2010[df_PRS_asthma_endpoint_2010['event_j45_3yr'] == 0]['SCORE'], df_PRS_asthma_endpoint_2010[df_PRS_asthma_endpoint_2010['event_j45_3yr'] == 1]['SCORE'])


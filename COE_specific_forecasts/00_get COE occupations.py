import pandas as pd
import os
basepath = 'C:/AnacondaProjects/CCC forecasting'
os.chdir(basepath)

df = pd.read_excel('emsi_skills_api/Academic Programs and Plans by COE - SU23.xlsx', skiprows=3)
df['COE'] = df['COE'].ffill()

df = df[['COE','SOC(s)']].drop_duplicates()
coe_socs = {}
for coe in df.COE.unique():
    if coe != 'Grand Total':
        coe_socs[coe] = []
        filt_df = df.loc[df.COE == coe]
        socs = filt_df['SOC(s)'].to_list()
        if 0 in socs:
            socs.remove(0)
        if 2634069 in socs:
            socs.remove(2634069)
        socs = [i.split(';') for i in socs]

        # flatten list
        socs = list(set([item for sublist in socs for item in sublist]))
        coe_socs[coe] = [i.replace('.','-') for i in socs]
results = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in coe_socs.items()]))
results.to_csv('emsi_skills_api/SOCS by COE.csv')
pass

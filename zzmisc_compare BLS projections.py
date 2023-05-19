import pandas as pd

bls_df = pd.read_csv('data/BLS projections/Employment Projections 2021.csv')
bls_df['occupation name'] = bls_df['Occupation Title'].str.split('*',expand=True)[0].str.strip().str.lower()
pred_df = pd.read_excel('output/exhibits/VAR_ARIMA ensemble formatted results.xlsx',sheet_name='Skill')
pred_df = pred_df.loc[pred_df['Monthly average obs'] > 500]
occ_df = pd.read_csv('output/most common occupations for each skill.csv')
occ_df = occ_df.iloc[:,0:2]
occ_df.columns = ['skill','occupation name']
pred_df = pred_df.merge(occ_df, on = 'skill')
pred_df['occupation name'] = pred_df['occupation name'].str.lower()
pred_df = pred_df.merge(bls_df[['occupation name','Employment Percent Change, 2021-2031']], on = 'occupation name', how='left')
print(pred_df[['Percent change','Employment Percent Change, 2021-2031']].corr())
pass

import pandas as pd

# check for seasonality.
df = pd.read_csv('data/test monthly counts.csv')
df = df.rename({'Unnamed: 0':'date'}, axis=1)
df['month']= df['date'].str[5:7].astype('int')
df = df.fillna(method='ffill')
# 7-55 filter is to remove months with 0 obs
df = df.iloc[7:55,:].reset_index(drop=True)

# create times series index
date_idx = pd.to_datetime(df['date'])

# normalize all columns based on job postings counts
df = df.drop('date', axis=1)
job_counts = df['Postings count'].copy()
raw_df = df.copy()
#df = df.divide(job_counts, axis=0)
df['Postings count'] = job_counts
df = df.set_index(pd.DatetimeIndex(date_idx))
targets = raw_df.mean(numeric_only=True).loc[raw_df.mean(numeric_only=True)<=50].index

exclude_df = df[targets].T

demand_diff = exclude_df.iloc[:,-1] - exclude_df.iloc[:,0]

demand_diff = demand_diff.sort_values(ascending = False)
demand_diff.name = 'Demand difference'
demand_diff = pd.DataFrame(demand_diff, index = exclude_df.index)
demand_diff = demand_diff.merge(exclude_df, left_index = True, right_index = True)
demand_diff = demand_diff.sort_values('Demand difference', ascending = False)
demand_diff.to_csv('working/demand difference of excluded skill.csv')

pass
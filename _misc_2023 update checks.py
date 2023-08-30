import pandas as pd
it = pd.read_csv('data/2023_update/AIR Datapull.csv')
#df = next(it)
df = it
df['POSTED'] = pd.to_datetime(df.POSTED).dt.date
df = df.loc[df.POSTED.isna() == False]
df['POSTED_MONTH'] = pd.to_datetime(df.POSTED).dt.month
df['POSTED_MONTH'] = df['POSTED_MONTH'].astype('int').round(0)
df['POSTED_YEAR'] = pd.to_datetime(df.POSTED).dt.year
df['POSTED_YEAR'] = df['POSTED_YEAR'].astype('int').round(0)

pass

# processing the NYT COVID county data file into monthly counts

import pandas as pd

df = pd.read_csv("data/NYT COVID us-counties.csv")

df = df.loc[df.fips==17031]
df['cases_change'] = df.cases.diff()
df = df.dropna()
df['month'] = df.date.str[5:7]
df['year'] = df.date.str[0:4]
covid_counts = df.groupby(['year','month']).sum()['cases_change']
# reweight last month with only partial data
covid_counts.loc[('2022','05')] = covid_counts.loc[('2022','05')] * 31 / 13
# Extrapolate to other months in the model
covid_counts.loc[('2022','06')] = covid_counts.loc[('2022','05')]
covid_counts.loc[('2022','07')] = covid_counts.loc[('2022','05')]

covid_counts.to_csv('data/NYT COVID us-counties clean.csv')
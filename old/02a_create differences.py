# build series that's the change in values over time

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import warnings
import numpy as np

df = pd.read_csv('data/test monthly counts season-adj.csv',index_col=0)
df.index = pd.to_datetime(df.index)

# drop columns with no postings (all sum to zero)
zero_sum_cols = df.columns[df.sum() == 0]
df = df.drop(zero_sum_cols,axis = 1)

targets = [i for i in df.columns if 'Skill:' in i]
result_df = pd.DataFrame(index=targets, columns=['ADF','KPSS'])

# track number of differences it takes to find a difference for each skill that is stationary
diff_dfs = {}
# take first order difference
diff_dfs[1] = df.diff()
# take second order difference
diff_dfs[2] = diff_dfs[1].diff()
# take third order difference
diff_dfs[3] = diff_dfs[2].diff()

diff_dfs[1] = diff_dfs[1].iloc[1:,:]
diff_dfs[2] = diff_dfs[2].iloc[2:,:]
diff_dfs[3] = diff_dfs[3].iloc[3:,:]

def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput['p-value']

def kpss_test(timeseries):
    with warnings.catch_warnings():
        # getting an interpolation warning when running kpss that we can ignore
        warnings.simplefilter('ignore')
        kpsstest = kpss(timeseries, regression='c', nlags="auto")
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
        return kpss_output['p-value']


diff_track = pd.Series(index= targets, name='differences taken')
# note which variables are non-stationary without differences
stationary_df = pd.read_csv('output/stationary test results.csv', index_col=0)
stationary_df = stationary_df.drop('Seasonality', axis = 1)
stationary_df = stationary_df.drop([i for i in stationary_df.index if 'Skill' not in i])

stationary_df = stationary_df.dropna()
stationary_df['stationary'] = stationary_df['ADF'] + stationary_df['KPSS']

stat_skills = stationary_df.loc[stationary_df.stationary == 0].index

diff_track.loc[stat_skills] = 0

for i in range(1,4):
    # only need to continue with variables that are non-stationary
    targets = diff_track.loc[diff_track.isna()].index

    print('check for stationarity of difference order ',i)
    diff_df = diff_dfs[i]
    for n,t in enumerate(targets):
        if n % 100 == 0:
            print(n ,'of',len(targets),'skills')
        if diff_df[t].sum() != 0:
            try:
                # record results of whether tests indicate non-stationarity
                adf_result = adf_test(diff_df[t]) < .01
                kpss_result = kpss_test(diff_df[t]) < .01

                if adf_result == False and kpss_result == False:
                    diff_track.loc[t] = i
            # running into some occasional errors with the kpss test, skip past those.
            except Exception as e:
                print(e)
                pass
    diff_df.to_csv("data/test monthly counts season-adj difference order"+str(i)+".csv")

diff_track.to_csv('working/stationary order by skill.csv')

# look at sample size of remaining skills still not stationary after 3rd order
samp_df = pd.read_csv('working/average monthly observations by counts.csv', index_col = 0)
comp_df = pd.concat([diff_track,samp_df], axis=1)
comp_df.columns = ['differences taken', 'monthly sample size']
means = comp_df.groupby('differences taken', dropna = False).mean()['monthly sample size']
pass
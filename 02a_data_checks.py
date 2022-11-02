# check for stationary trends in skills data

from multivar_forecast_loop import prepare_data
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot

df, targets = prepare_data()
targets = targets[1:]
result_df = pd.DataFrame(index=targets, columns=['ADF','KPSS','Seasonality'])
df.index = pd.to_datetime(df.index)
job_counts = df['Postings count'].copy()
df = df.divide(job_counts, axis=0)

def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput['p-value']

def kpss_test(timeseries):
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    return kpss_output['p-value']

for n,t in enumerate(targets):
    if n % 100 == 0:
        print(n ,'of',len(targets),'skills')

    # record results of whether tests indicate non-stationarity
    result_df.loc[t,'ADF'] = adf_test(df[t]) < .05
    result_df.loc[t,'KPSS'] = kpss_test(df[t]) < .05

result_df.to_csv('output/stationary test results.csv')


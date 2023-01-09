# check for stationary trends in skills data

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels

df = pd.read_csv('data/test monthly counts season-adj.csv',index_col=0)
targets = df.columns[1:]
df.index = pd.to_datetime(df.index)
# check causation using granger's causality test
maxlag=5
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            try:
                test_result = grangercausalitytests(data[[r, c]], maxlag=5, verbose=False)
                p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
            except statsmodels.tools.sm_exceptions.InfeasibleTestError:
                print('infeasible test for ', c, 'and',r)
                min_p_value = np.nan

            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

cause_df = grangers_causation_matrix(df[:20], variables = df[:20].columns)
cause_df.to_csv('working/granger_causation_results.csv')
pass

result_df = pd.DataFrame(index=targets, columns=['ADF','KPSS','Seasonality'])

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
    if sum(df[t]) >0:
        result_df.loc[t,'ADF'] = adf_test(df[t]) > .05
        result_df.loc[t,'KPSS'] = kpss_test(df[t]) > .05
    else:
        result_df.loc[t, 'ADF'] = np.nan
        result_df.loc[t, 'KPSS'] = np.nan

print(result_df.ADF.value_counts())
print(result_df.KPSS.value_counts())

result_df.to_csv('output/stationary test results.csv')


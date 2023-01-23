from darts import TimeSeries, concatenate
import darts.metrics
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels
import matplotlib.pyplot as plt

def grangers_causation_matrix(data, variables, target, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    target    : target variable model is looking to estimate
    test      : type of test to perform
    verbose   : whether to toggle verbose log printing
    """
    maxlag = 5
    df = pd.DataFrame(np.zeros((len(variables), 1)), columns=[target], index=variables)
    for r in df.index:
        try:
            test_result = grangercausalitytests(data[[r, target]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {target}, P Values = {p_values}')
            min_p_value = np.min(p_values)
        except statsmodels.tools.sm_exceptions.InfeasibleTestError:
            print('infeasible test for ', target, 'and',r)
            min_p_value = np.nan
        except ValueError:
            print('infeasible test for ', target, 'and', r)
            min_p_value = np.nan

        df.loc[r, target] = min_p_value
    return df

def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    #print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    result = pd.Series(index=df.columns)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        #print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        result.loc[col] = trace > cvt
    return result

def adf_test(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput['p-value']

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax
    return ({'mape': mape, 'me': me, 'mae': mae,
             'mpe': mpe, 'rmse': rmse, 'corr': corr, 'minmax': minmax})


def invert_transformation(df_train, df_forecast, diffs_made):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        diff_col = df_fc[str(col) + '_' + str(diffs_made) + 'd']
        diffs_made_temp = diffs_made
        while diffs_made_temp >= 0:
            # Roll back each level of diff
            init_val = df_train[col].iloc[-1]
            for i in range(-1, (diffs_made_temp*-1)-1, -1):
                init_val -= df_train[col].iloc[i]
            diff_col = init_val + diff_col.cumsum()
            diffs_made_temp -= 1
        df_fc[str(col) + '_forecast'] = diff_col
    return df_fc

# helper function: get forecast values for selected quantile q and insert them in dataframe dfY
def predQ(ts_t, q, scalerP, dfY, ts_test, quantile=True):
    if quantile:
        ts_tq = ts_t.quantile_timeseries(q)
        ts_q = scalerP.inverse_transform(ts_tq)
        s = TimeSeries.pd_series(ts_q)
        header = "Q" + format(int(q * 100), "02d")
        dfY[header] = s
        if q == 0.5:
            ts_q50 = ts_q
            q50_RMSE = darts.metrics.rmse(ts_q50, ts_test)
            q50_MAPE = darts.metrics.mape(ts_q50, ts_test)
            print("RMSE:", f'{q50_RMSE:.2f}')
            print("MAPE:", f'{q50_MAPE:.2f}')
            return [q50_RMSE, q50_MAPE]
    else:
        ts_tq = ts_t
        ts_q = scalerP.inverse_transform(ts_tq)
        s = TimeSeries.pd_series(ts_q)
        ts_q50 = ts_q
        q50_RMSE = darts.metrics.rmse(ts_q50, ts_test)
        q50_MAPE = darts.metrics.mape(ts_q50, ts_test)
        print("RMSE:", f'{q50_RMSE:.2f}')
        print("MAPE:", f'{q50_MAPE:.2f}')
        return [q50_RMSE, q50_MAPE]

# Analyze results of predictions
def results_analysis(fcast_filename):
    # verify we're using a model output file
    assert('predicted job posting shares' in fcast_filename)
    df = pd.read_csv('output/' + fcast_filename+'.csv', index_col=0)
    countdf = pd.read_csv('data/test monthly counts season-adj.csv', index_col=0)

    raw_df = pd.read_csv("data/test monthly counts.csv", index_col=0)
    raw_df = raw_df.fillna(method='ffill')
    raw_df = raw_df.iloc[7:55, :]

    # clean up some duplicated columns in the forecast
    df = df.drop([i for i in df.columns if '1' in i], axis=1)
    if 'covid_cases' in df.columns:
        df = df.drop("covid_cases", axis=1)
    targets = df.columns

    countdf = countdf[targets]

    lactual = countdf.iloc[-1, :]
    pred_values = df.iloc[-3, :]

    change = ((pred_values - lactual) / lactual) * 100

    result_df = pd.concat([lactual, pred_values, change], axis=1)

    result_df.columns = ['July 2022 actual', 'July 2024 predicted', 'Percent change']

    result_df = result_df.sort_values('Percent change', ascending=False)

    # add sample size to the results
    month_counts = raw_df.mean()
    month_counts.name = 'Monthly average obs'
    result_df = result_df.merge(month_counts, left_index= True, right_index=True)

    result_df.index = [i.replace('Skill: ', '') for i in result_df.index]

    result_df.to_csv('output/predicted changes '+ fcast_filename.replace('predicted job posting shares ','')+'.csv')

def forecast_graph(pred, actual, label, folder):
    pred.name = 'Predicted'
    actual.name = 'Actual'

    pred.plot.line()
    actual.plot.line()
    plt.legend()
    plt.title(label)
    plt.savefig(folder+'/'+label+'.png')
    plt.clf()

def visualize_predictions(fcast_filename, sample = 10):
    run_name = fcast_filename.replace('predicted job posting shares ','')
    pred_df = pd.read_csv("output/"+fcast_filename+".csv", index_col=0)
    act_df = pd.read_csv('data/test monthly counts season-adj.csv', index_col=0)
    pred_df.index = pd.to_datetime(pred_df.index)
    act_df.index = pd.to_datetime(act_df.index)

    #job_counts = act_df['Postings count'].copy()
    #act_df = act_df.divide(job_counts, axis=0)
    #act_df['Postings count'] = job_counts

    if not os.path.exists('output/exhibits/'+run_name):
        os.mkdir('output/exhibits/'+run_name)

    for i in pred_df.columns[:sample]:
        if i != 'Postings count':
            forecast_graph(pred_df[i], act_df[i], i.replace('Skill: ','')+' graph', 'output/exhibits/'+run_name)

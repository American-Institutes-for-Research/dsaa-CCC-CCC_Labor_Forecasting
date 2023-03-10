'''
Luke Patterson
utils.py

Purpose: define various utility functions used throughout other code files

'''

import glob
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
import shutil


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
            print('infeasible test for ', target, 'and', r)
            min_p_value = np.nan
        except ValueError:
            print('infeasible test for ', target, 'and', r)
            min_p_value = np.nan

        df.loc[r, target] = min_p_value
    return df


def cointegration_test(df, alpha=0.05):
    '''
    Perform Johanson's Cointegration Test and Report Summary from dataframe
    :param df: dataframe of variables to test for cointegration
    :param alpha: int, alpha value of test
    :return:
    Series, Whether test result is significant for each column
    '''

    out = coint_johansen(df, -1, 5)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]
    # def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    # print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    result = pd.Series(index=df.columns)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        # print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        result.loc[col] = trace > cvt
    return result


def adf_test(timeseries):
    '''
    Perform Augmented Dickey-Fuller test on a series
    :param timeseries: array-like, Time series to test
    :return:
    int, p-value of test
    '''
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput['p-value']


def forecast_accuracy(forecast, actual):
    '''
    Calculate forecast performance metrics
    :param forecast: Array-like, forecast values
    :param actual: Array-like, actual values
    :return: dict of forecast performance metrics
    '''
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
    '''
    Revert back the differencing to get the forecast to original scale.
    :param df_train: DataFrame of training data
    :param df_forecast: DataFrame of forecasts made
    :param diffs_made: number of differences taken
    :return: transformed df_forecast with reverted differencing
    '''
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        diff_col = df_fc[str(col) + '_' + str(diffs_made) + 'd']
        diffs_made_temp = diffs_made
        while diffs_made_temp >= 0:
            # Roll back each level of diff
            init_val = df_train[col].iloc[-1]
            for i in range(-1, (diffs_made_temp * -1) - 1, -1):
                init_val -= df_train[col].iloc[i]
            diff_col = init_val + diff_col.cumsum()
            diffs_made_temp -= 1
        df_fc[str(col) + '_forecast'] = diff_col
    return df_fc


def predQ(ts_t, q, scalerP, ts_test, quantile=True):
    '''
    helper function: get forecast values for selected quantile q
    :param ts_t: darts TimeSeries, predicted values matching test data set months
    :param q: int, quantile to get values for
    :param scalerP: Scalar object of ts_t
    :param ts_test: darts TimeSeries, actual values from test data set
    :param quantile: quantile to get forecast values for
    :return: list, [RMSE, MAPE] of series
    '''
    if quantile:
        ts_tq = ts_t.quantile_timeseries(q)
        ts_q = scalerP.inverse_transform(ts_tq)
        if q == 0.5:
            ts_q50 = ts_q
            q50_RMSE = darts.metrics.rmse(ts_q50, ts_test)
            q50_MAPE = darts.metrics.mape(ts_q50, ts_test)
            print("RMSE:", f'{q50_RMSE:.2f}')
            print("MAPE:", f'{q50_MAPE:.2f}')
            return [q50_RMSE, q50_MAPE]
    else:
        ts_q = scalerP.inverse_transform(ts_t)
        q50_RMSE = darts.metrics.rmse(ts_q, ts_test)
        q50_MAPE = darts.metrics.mape(ts_q, ts_test)
        print("RMSE:", f'{q50_RMSE:.2f}')
        print("MAPE:", f'{q50_MAPE:.2f}')
        return [q50_RMSE, q50_MAPE]


# Analyze results of predictions
def results_analysis(fcast_filename, create_vizs=False, panel_data = False):
    '''
    Analyze the results of a particular forecast run
    :param fcast_filename: str, filename of the forecasts to analyze
    :param create_vizs: bool, whether to create visualizations of the run's performance
    :return:
    None
    '''
    # verify we're using a model output file
    assert ('predicted job posting shares' in fcast_filename)
    df = pd.read_csv('output/' + fcast_filename + '.csv', index_col=0)
    df = df.rename({'Unnamed: 1': 'month'}, axis=1)
    if panel_data:
        df = agg_panel_data(df)

    if 'lvl subcategory' in fcast_filename:
        countdf = pd.read_csv('data/wrong counts/test monthly counts season-adj subcategory.csv', index_col=0)
        raw_df = pd.read_csv("data/wrong counts/test monthly counts categories.csv", index_col=0)
        raw_df = raw_df[[i for i in raw_df.columns if 'Skill cat:' not in i]]

    elif 'lvl category' in fcast_filename:
        countdf = pd.read_csv('data/wrong counts/test monthly counts season-adj category.csv', index_col=0)
        raw_df = pd.read_csv("data/wrong counts/test monthly counts categories.csv", index_col=0)
        raw_df = raw_df[[i for i in raw_df.columns if 'Skill subcat:' not in i]]

    else:
        countdf = pd.read_csv('data/wrong counts/test monthly counts season-adj skill.csv', index_col=0)
        raw_df = pd.read_csv("data/wrong counts/test monthly counts.csv", index_col=0)

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
    result_df = result_df.merge(month_counts, left_index=True, right_index=True)

    result_df.index = [i.replace('Skill: ', '') for i in result_df.index]

    result_df.to_csv(
        'output/predicted changes/predicted changes ' + fcast_filename.replace('predicted job posting shares ',
                                                                               '') + '.csv')

    if create_vizs:
        visualize_predictions(fcast_filename)


def forecast_graph(preds, actual, col, label, folder):
    '''
    Create graph comparing prediction model(s) to actual data
    :param preds: dict of DataFrames, keys are labels of prediction models, values are Series of different prediction model
        outputs to graph
    :param actual: DataFrames, actual data to compare
    :param col: string, name of column to model
    :param label: string, title/filename to call
    :param folder: string, folder to save
    :return:
    None
    '''

    for key, pred in zip(preds.keys(), preds.values()):
        pred = pred[col]
        pred.name = key
        pred.plot.line()

    label = label.replace('/', '_')
    actual = actual[col]
    actual.name = 'Actual'
    actual.plot.line()
    plt.legend()
    plt.title(label)
    plt.savefig(folder + '/' + label + '.png')
    plt.clf()


def visualize_predictions(fcast_filename=None, sample=10, topfiles=None, panel_data=False, model_name='Predicted'):
    '''
    Visualize the predictions against the actual data
    :param fcast_filename: string, name of data with the forecasts
    :param sample: int, sample of predicted variables to take
    :param topfiles: None or int, number of most recent files in output folder to make predictions for
    :param panel_data: bool, whether data is panel or not
    :return:
    None
    '''
    if fcast_filename is not None:
        filenames = ['output\\' + fcast_filename + '.csv']

    if topfiles is not None:
        search_dir = "output/"
        # remove anything from the list that is not a file (directories, symlinks)
        # of files (presumably not including directories)
        files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        filenames = files[0:topfiles]

    if fcast_filename is None and topfiles is None:
        raise ('must specify either fcast_filename or topfiles')

    for name in filenames:
        run_name = name.replace('predicted job posting shares ', '')
        run_name = run_name.replace('output\\', '')
        run_name = run_name.replace('.csv', '')

        pred_df = pd.read_csv(name, index_col=0)
        pred_df = clean_data_for_graphs(pred_df, run_name, panel_data=panel_data)

        if 'lvl subcategory' in run_name:
            act_df = pd.read_csv('data/wrong counts/test monthly counts season-adj subcategory.csv', index_col=0)
        elif 'lvl category' in run_name:
            act_df = pd.read_csv('data/wrong counts/test monthly counts season-adj category.csv', index_col=0)
        else:
            act_df = pd.read_csv('data/wrong counts/test monthly counts season-adj skill.csv', index_col=0)

        act_df.index = pd.to_datetime(act_df.index)

        # job_counts = act_df['Postings count'].copy()
        # act_df = act_df.divide(job_counts, axis=0)
        # act_df['Postings count'] = job_counts

        if not os.path.exists('output/exhibits/' + run_name):
            os.mkdir('output/exhibits/' + run_name)

        if sample == None:
            pred_cols = pred_df.columns
        else:
            pred_cols = pred_df.columns[:sample]

        for i in pred_cols:
            if i != 'Postings count':
                skill_name = i.replace('Skill: ', '').replace('Skill cat:', '').replace('Skill subcat:', '')
                forecast_graph({model_name: pred_df}, act_df, i, skill_name + ' graph', 'output/exhibits/' + run_name)


def clean_data_for_graphs(pred_df, run_name, panel_data=False):
    '''
    read
    :param name: string with filename of predictions
    :param panel_data: whether data is panel by county or not
    :return:
    pred_df: dataframe of predictions
    '''

    if panel_data:
        pred_df = pred_df.rename({'Unnamed: 1': 'month'}, axis=1)
        pred_df = agg_panel_data(pred_df)

    pred_df.index = pd.to_datetime(pred_df.index)

    return pred_df


def agg_panel_data(df):
    '''
    aggregate counties from panel data to get overall estimates for Chicago MSA

    :param
     df: dataframe of postings share predictions made by the panel model by county

    :return:
    DataFrame of aggregated postings share predictions for overall Chicago MSA
    '''

    # load data for number of postings occuring in each county
    # count_df = pd.read_csv('data/test monthly counts county panel.csv')
    # count_df = count_df.set_index('Unnamed: 0')
    # count_df['Postings count'].to_csv('data/county panel postings sample size.csv')
    count_df = pd.read_csv('data/county panel postings sample size.csv', index_col=0)

    # we will weight the posting shares by average postings for each county
    count_df.index = count_df.index.map(lambda idx: idx.split("'")[1])
    count_df = count_df.reset_index().rename({'Unnamed: 0': 'county'}, axis=1)
    count_df = count_df.groupby('county').mean()
    count_df = count_df / count_df.sum()
    # there are two Lake counties, so divide that weight by half. Technically should weight separately, but the
    # counties are close enough in size that it is not a big difference.
    count_df.loc['Lake'] = count_df.loc['Lake'] / 2

    # cut to only county name
    df.index = df.index.map(lambda idx: idx.split(",")[0])

    df = df.merge(count_df, left_index=True, right_index=True)

    # apply weights to columns
    values = [i for i in df.columns if i not in ['month', 'Postings count']]
    df = df.reset_index(drop=True)
    for c in values:
        df[c] = df[c].multiply(df['Postings count'])

    result_df = df.groupby('month').sum()
    result_df = result_df.drop('Postings count', axis=1)
    return result_df


def compare_results(runnames, labels, title, panel_indicators, hierarchy_lvl='skill', sample=None):
    '''
    Produce comparisons of two or more sets of results
    :params:
    runnames: list of filenames of the runs of results to use for comparisons
    labels: list of labels to use as shorthand for runnames in charts. Should be same length as runnames. Should take values: 'panel', "VAR", 'transformer'
    title: string, title to give the comparison
    panel_indicators: list of booleans indicating whether run files are panels or not
    :return:
    None
    '''
    valid_labels = ['panel', "VAR", 'transformer']
    assert all([i in valid_labels for i in labels]),'all labels must be one of panel, VAR, transformer'
    if hierarchy_lvl == 'subcategory':
        act_df = pd.read_csv('data/wrong counts/test monthly counts season-adj subcategory.csv', index_col=0)
    elif hierarchy_lvl == 'category':
        act_df = pd.read_csv('data/wrong counts/test monthly counts season-adj category.csv', index_col=0)
    else:
        act_df = pd.read_csv('data/wrong counts/test monthly counts season-adj skill.csv', index_col=0)

    act_df.index = pd.to_datetime(act_df.index)

    dfs = {}
    cols = []
    for n, l in enumerate(labels):
        dfs[l] = pd.read_csv('output/' + runnames[n] + '.csv', index_col=0)
        if panel_indicators[n]:
            dfs[l] = dfs[l].rename({'Unnamed: 1': 'month'}, axis=1)
            dfs[l] = agg_panel_data(dfs[l])

        dfs[l].index = pd.DatetimeIndex(dfs[l].index)
        # establish the set of columns that all data sets share
        if n == 0:
            cols = dfs[l].columns
        else:
            cols = [i for i in cols if i in dfs[l].columns.values]

    # produce visualizations with all runs on the same diagram
    if sample == None:
        cols = cols
    else:
        cols = cols[:sample]

    # create graphs for each column
    if os.path.exists('output/exhibits/' + title):
        shutil.rmtree('output/exhibits/' + title)
    os.mkdir('output/exhibits/' + title)

    for c in cols:
        skill_name = c.replace('Skill: ', '').replace('Skill cat:', '').replace('Skill subcat:', '')
        forecast_graph(dfs, act_df, c, skill_name + ' graph', 'output/exhibits/' + title)

    # compare performance metrics
    # this dataframe keeps track of mean overall metrics by model, and then appends the individual model metrics at the end
    perf_df = pd.DataFrame(index=['mean'])
    log_dfs = {}
    for n, l in enumerate(labels):
        log_name = runnames[n].replace('predicted job posting shares','looped '+labels[n]+ ' model results')
        log_df = pd.read_csv('result_logs/' + log_name + '.csv', index_col=0)
        log_dfs[l] = log_df
        # some of the older logs need to be inverted
        if 'MAPE' not in log_df.columns:
            log_df = log_df.T
        log_df = log_df.set_index('target')
        # if it's panel, metrics are recorded at the target level, so we can drop duplicates from counties
        if panel_indicators[n]:
            log_df = log_df[~log_df.index.duplicated(keep='first')]

        # add the model's mean performance over all targets
        perf_df.loc['mean', 'Normalized RMSE '+labels[n]] = log_df['Normalized RMSE'].astype('float').mean()
        perf_df.loc['mean','MAPE '+labels[n]] = log_df['MAPE'].astype('float').mean()

        log_df = log_df.rename({'Normalized RMSE':'Normalized RMSE '+labels[n],'MAPE':'MAPE '+labels[n]}, axis=1)
        # for first loop we need to add the axis values
        if n == 0:
            perf_df = pd.concat([perf_df, log_df[['Normalized RMSE '+labels[n],'MAPE '+labels[n]]]])

        # for the rest, we can just merge on axis values
        else:
            perf_df.update(log_df[['Normalized RMSE '+labels[n],'MAPE '+labels[n]]])


    perf_df.to_excel('output/exhibits/'+title+'/performance comparison.xlsx')

    # output statistics on agreement of models in terms of skill demand rankings
    pred_dfs = {}
    # Load each data frame of predicted changes and rename columns so they can be merged together
    for label, runname in zip(labels, runnames):
        pred_name = runname.replace('predicted job posting shares','predicted changes')
        df = pd.read_csv('output/predicted changes/'+pred_name+'.csv')
        df = df.set_index('Unnamed: 0', drop = True)
        # these columns should be the same across all, so we'll add at the end
        actual = df['July 2022 actual']
        avg_obs = df['Monthly average obs']
        df = df.drop(['Monthly average obs','July 2022 actual'], axis = 1)
        df = df.sort_values('Percent change', ascending= False)
        df['% change rank'] = np.arange(df.shape[0])+1
        df.columns = [label+'_'+i for i in df.columns]
        pred_dfs[label] = df

    merge_df = pd.concat(pred_dfs.values(), axis = 1)
    merge_df['Monthly average obs'] = avg_obs
    merge_df['July 2022 Actual'] = actual

    # add change from July 2021 to July 2022 for comparison
    merge_df['Aug 2018 Actual'] = act_df.loc[pd.to_datetime('2018-08-01')]
    merge_df['18-22 Actual_Percent change'] = ((merge_df['July 2022 Actual'] - merge_df['Aug 2018 Actual']) / merge_df['Aug 2018 Actual']) * 100
    merge_df = merge_df.sort_values('18-22 Actual_Percent change', ascending=False)
    merge_df['18-22 Actual_% change rank'] = np.arange(merge_df.shape[0]) + 1

    # reorder some vars for readibility
    merge_df = merge_df[
        [i for i in merge_df.columns if '% change rank' in i] +
        [i for i in merge_df.columns if 'Percent change' in i] +
        [i for i in merge_df.columns if 'July 2024 predicted' in i] +
        ['Aug 2018 Actual','July 2022 Actual', 'Monthly average obs']
    ]

    merge_df.to_excel('output/exhibits/' + title + '/skill ranking comparison full.xlsx')
    
    merge_df = merge_df.loc[merge_df['Monthly average obs'] > 1000]
    labels.append('18-22 Actual')
    for l in labels:
        merge_df = merge_df.sort_values(l+'_Percent change', ascending = False)
        merge_df[l+'_% change rank'] = np.arange(merge_df.shape[0])+1
    merge_df.to_excel('output/exhibits/' + title + '/skill ranking comparison over 1000 avg obs.xlsx')

    print(merge_df.iloc[:,:4].corr())


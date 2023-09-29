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
import itertools
from statsmodels.stats.weightstats import DescrStatsW

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

def invert_transformation_series(df_train, forecast_series, diffs_made):
    '''
       Revert back the differencing to get the forecast to original scale for a series.
       :param df_train: DataFrame of training data
       :param df_forecast: Series of forecast made
       :param diffs_made: number of differences taken
       :return: transformed series forecast with reverted differencing
       '''
    diffs_made_temp = diffs_made
    result = forecast_series.copy()
    col = forecast_series.name
    while diffs_made_temp >= 0:
        # Roll back each level of diff
        init_val = df_train[col].iloc[-1]
        for i in range(-1, (diffs_made_temp * -1) - 1, -1):
            init_val -= df_train[col].iloc[i]
        result = init_val + result.cumsum()
        diffs_made_temp -= 1
    return result

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
def results_analysis(fcast_filename, fcast_folder = 'output/', create_vizs=False, panel_data = False, countdf=None, raw_df=None,
                     act_varname = 'July 2022 actual', pred_varname = 'July 2024 predicted' ):
    '''
    Analyze the results of a particular forecast run
    :param fcast_filename: str, filename of the forecasts to analyze
    :param create_vizs: bool, whether to create visualizations of the run's performance
    :return:
    None
    '''
    # verify we're using a model output file
    assert ('predicted job posting shares' in fcast_filename)
    df = pd.read_csv(fcast_folder + fcast_filename + '.csv', index_col=0)
    df = df.rename({'Unnamed: 1': 'month'}, axis=1)
    if panel_data:
        df = agg_panel_data(df)

    # load the season adjusted and non seasonally adjusted counts files if not fed as parameters
    if raw_df is None:
        if 'lvl subcategory' in fcast_filename:
            raw_df = pd.read_csv("data/test monthly counts categories 2023 update.csv", index_col=0)
            raw_df = raw_df[[i for i in raw_df.columns if 'Skill cat:' not in i]]

        elif 'lvl category' in fcast_filename:
            raw_df = pd.read_csv("data/test monthly counts categories 2023 update.csv", index_col=0)
            raw_df = raw_df[[i for i in raw_df.columns if 'Skill subcat:' not in i]]

        else:
            raw_df = pd.read_csv("data/test monthly counts 2023 update.csv", index_col=0)

        raw_df = raw_df.fillna(method='ffill')
        # 2022 files
        if raw_df.shape[0] == 60:
            raw_df = raw_df.iloc[7:67, :]
        # 2023 files
        elif raw_df.shape[0] == 72:
            raw_df = raw_df.iloc[7:67, :]
        else:
            raise('expected raw_df to have either 60 or 72 rows')

    if countdf is None:
        if 'lvl subcategory' in fcast_filename:
            countdf = pd.read_csv('data/test monthly counts season-adj subcategory.csv', index_col=0)

        elif 'lvl category' in fcast_filename:
            countdf = pd.read_csv('data/test monthly counts season-adj category.csv', index_col=0)

        else:
            countdf = pd.read_csv('data/test monthly counts season-adj skill.csv', index_col=0)

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

    result_df.columns = [act_varname, pred_varname, 'Percent change']

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
    plt.clf()
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
            act_df = pd.read_csv('data/test monthly counts season-adj subcategory.csv', index_col=0)
        elif 'lvl category' in run_name:
            act_df = pd.read_csv('data/test monthly counts season-adj category.csv', index_col=0)
        else:
            act_df = pd.read_csv('data/test monthly counts season-adj skill.csv', index_col=0)

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
                skill_name = i.replace('Skill: ', '').replace('Skill cat: ', '').replace('Skill subcat: ', '')
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

    # cut eto only county name
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
    labels: list of labels to use as shorthand for runnames in charts. Should be same length as runnames. Should take values: 'panel', "VAR", 'transformer','ProphetAR','ARIMA'
    title: string, title to give the comparison
    panel_indicators: list of booleans indicating whether run files are panels or not
    :return:
    None
    '''
    valid_labels = ['panel', "VAR", 'transformer','ProphetAR','ARIMA']
    #assert all([i in valid_labels for i in labels]),'all labels must be one of panel, VAR, transformer'
    if hierarchy_lvl == 'subcategory':
        act_df = pd.read_csv('data/test monthly counts season-adj subcategory.csv', index_col=0)
    elif hierarchy_lvl == 'category':
        act_df = pd.read_csv('data/test monthly counts season-adj category.csv', index_col=0)
    else:
        act_df = pd.read_csv('data/test monthly counts season-adj skill.csv', index_col=0)

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
    print('loaded data')
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
    print('finished graphing comparison forecasts')
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
    print('completed performance metric comparison')
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

    merge_df.corr().to_csv('output/exhibits/' + title + '/model skill ranking correlation.csv')


def grid_search(params_grid, default_params, loop_func, batch_name, clear = True):
    '''
    Perform a grid search over combinations of parameters in param grid to find
    :param params_grid: dict of lists representing possible parameters to search
    :param default_params: default params to use in every run
    :param loop_func: function to loop over
    :param batch_name: name of batch
    :param clear: whether to clear folder if it exists
    :return: None
    '''

    # remove folder from previous run if present
    if clear:
        if os.path.exists('result_logs/batch_'+batch_name):
            shutil.rmtree('result_logs/batch_'+batch_name)
        if os.path.exists('output/batch_'+batch_name):
            shutil.rmtree('output/batch_'+batch_name)

    # create all combinations of params
    keys, values = zip(*params_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    error_tracker = pd.DataFrame()

    # iterate through all params for the model loop
    for n, p in enumerate(permutations_dicts):
        print('iteration', n, 'of', len(permutations_dicts))
        try:
            loop_func(run_name = batch_name, batch_name=batch_name, analyze_results = False, **default_params, **p)
        except Exception as e:
            print('error with params:', p)
            print('error message:', e)
            err_row = pd.Series(p)
            err_row['error'] = e
            error_tracker = error_tracker.append(err_row, ignore_index=True)
            continue

    error_tracker.to_csv('result_logs/batch_'+ batch_name +'/error_tracker.csv')

    logfiles = os.listdir('result_logs/batch_'+batch_name)
    logfiles = [i for i in logfiles if 'looped' in i and 'model results' in i]

    # loop through all log files and output a sorted list of the model iteration RMSE
    result_df = pd.DataFrame()
    for f in logfiles:
        df = pd.read_csv('result_logs/batch_' + batch_name + '/' + f, index_col=0)
        row = pd.Series()
        if 'Normalized RMSE' in df.columns and df.shape[0] > 1:
            row['filename'] = f
            row['RMSE'] = df['Normalized RMSE'].mean()
            for col in params_grid.keys():
                row[col] = df[col].iloc[0]
            if 'cand_features_num' in df.columns:
                row['cand_features_num'] = df['cand_features_num'].iloc[0]
            result_df = result_df.append(row, ignore_index=True)

    result_df = result_df.sort_values('RMSE')
    result_df.to_csv('result_logs/batch_'+ batch_name+'/RMSE summary.csv')

def create_ensemble_results(title, runnames= None, runnames_folders=None, panel_indicators = None, labels=None, types = None,
                            batch_names = [],  extreme_change_thresh = 1000, min_monthly_obs = 200, hierarchy_lvl='skill',
                            model_selection = 'best', output_occ_codes = False, do_results_analysis=False, use_coe_fcast_folder = False,
                            log_folder = 'result_logs/', act_df = None, raw_df = None, rerun_2023 = True):
    '''
    Produce combined ensemble estimates from two or more sets of results, choosing the lowest RMSE estimate for each skill
    :params:
    runnames: list of filenames of the runs of results to use for ensemble
    runnames: list of folders containing the runs of results to use for ensemble
    labels: list of labels to use as shorthand for runnames in charts. Should be same length as runnames. Should take values: 'panel', "VAR", 'transformer','ProphetAR', 'ARIMA']
    title: string, title to give the ensemble
    panel_indicators: list of booleans indicating whether run files are panels or not
    :return:
    None
    '''

    if types:
        valid_types = ['panel', "VAR", 'transformer', 'ProphetAR', 'ARIMA']
        assert all([i in valid_types for i in types]), 'all labels must be one of panel, VAR, transformer'
    if act_df is None:
        if hierarchy_lvl == 'subcategory':
            act_df = pd.read_csv('data/test monthly counts season-adj subcategory.csv', index_col=0)
        elif hierarchy_lvl == 'category':
            act_df = pd.read_csv('data/test monthly counts season-adj category.csv', index_col=0)
        else:
            act_df = pd.read_csv('data/test monthly counts season-adj skill.csv', index_col=0)

    act_df.index = pd.to_datetime(act_df.index)

    dfs = {}
    log_dfs = {}
    cols = []
    rmse_compare = pd.DataFrame()
    if rerun_2023:
        act_varname = 'July 2023 actual'
        pred_varname = 'July 2025 predicted'
    else:
        act_varname = 'July 2022 actual'
        pred_varname = 'July 2024 predicted'

    for n, l in enumerate(labels):
        if do_results_analysis:
            if use_coe_fcast_folder:
                if 'VAR' in l:
                    fcast_folder = 'output/batch_COE VAR runs v2/'
                if 'ARIMA' in l:
                    fcast_folder = 'output/batch_COE ARIMA runs v2/'
            else:
                if 'VAR' in l:
                    fcast_folder = 'output/batch_VAR top grid search runs 2023 rerun/'
                if 'ARIMA' in l:
                    fcast_folder = 'output/batch_ARIMA top grid search runs 2023/'

            results_analysis(fcast_filename=runnames[n], fcast_folder=fcast_folder, countdf = act_df, raw_df = raw_df, act_varname=act_varname, pred_varname=pred_varname)
        pred_df = pd.read_csv('output/predicted changes/' + runnames[n].replace('predicted job posting shares','predicted changes') + '.csv', index_col=0)

        if panel_indicators[n]:
            pred_df = pred_df.rename({'Unnamed: 1': 'month'}, axis=1)
            pred_df = agg_panel_data(pred_df)

        # establish the set of columns that all data sets share
        if n == 0:
            cols = pred_df.columns
        else:
            cols = [i for i in cols if i in pred_df.columns.values]

        dfs[runnames[n]] = pred_df
        log_name = runnames[n].replace('predicted job posting shares', 'looped ' + types[n] + ' model results')

        # if part of batches, identify the log folder the log can be found in
        if batch_names:
            found = False
            for folder in batch_names:
                if os.path.exists('result_logs/'+folder+'/'+ log_name+'.csv'):
                    log_folder = 'result_logs/'+folder + '/'
                    found = True
                    break
            assert found, 'batch log folder not found'
        else:
            log_folder = log_folder



        log_df = pd.read_csv(log_folder + log_name + '.csv', index_col=0)
        # some of the older logs need to be inverted
        if 'MAPE' not in log_df.columns:
            log_df = log_df.T
        log_df['target'] = log_df.target.str.replace('Skill: ', '')
        log_df = log_df.set_index('target')
        # if it's panel, metrics are recorded at the target level, so we can drop duplicates from counties
        if panel_indicators[n]:
            log_df = log_df[~log_df.index.duplicated(keep='first')]

        log_df = log_df.rename({'Normalized RMSE':'RMSE model #'+str(runnames[n])}, axis = 1)
        if n == 0:
            rmse_compare = log_df[['RMSE model #'+str(runnames[n])]]
        else:
            rmse_compare = rmse_compare.merge(log_df[['RMSE model #'+str(runnames[n])]], left_index=True, right_index=True, how='outer')
        log_dfs[runnames[n]] = log_df


    # sometimes models could score strong RMSE, but predict  changes far out of bounds of reality (as in > 1 or <0 job posting share).
    # also removing extreme change threshold
    # here we will remove these model's estimates from consideration for those skills

    for model, df in zip(dfs.keys(), dfs.values()):
        err_df = df.loc[(df[pred_varname] > 1) | (df[pred_varname] < 0) | (df['Percent change'].abs() > extreme_change_thresh)]
        print('for hierarchy level:',hierarchy_lvl, 'and model:',model)
        print(err_df.shape[0], ' predictions were too extreme or out of the [0,1] domain, and were removed from model candidates')

        for skill in err_df.index:
            rmse_compare.loc[skill, 'RMSE model #'+model] = np.nan


    # drop rows for which no model makes a valid prediction
    rmse_compare = rmse_compare.dropna(how='all')

    # option to select model with lowest RMSE
    if model_selection == 'best':
        best_models = rmse_compare.idxmin(axis=1).dropna()
        if hierarchy_lvl == 'skill':
            best_models.index = [i.replace('Skill: ', '') for i in best_models.index]

        ensemble_df = pd.DataFrame()
        missing_count = 0
        for skill,model in best_models.iteritems():
            model_label = model.replace('RMSE model #','')

            # handful of skills are in the log dfs but not in the results dfs due to model  failures:
            if skill in dfs[model_label].index:
                row = dfs[model_label].loc[skill,:]
                row['model'] = model_label
                row['Normalized RMSE'] = rmse_compare.loc[skill,model]
                ensemble_df = pd.concat([ensemble_df, row], axis = 1)
            else:
                missing_count += 1
        print(ensemble_df.shape[1],'skills added to ensemble df.', missing_count, 'skills dropped due to lack of predictions made.')
        ensemble_df = ensemble_df.T

    # option to take average of all models, weighted by RMSE
    if model_selection == 'weighted average':

        # create a weighted df that's a min max transformation of the inverse of the RMSE value of the model
        wgt_df = 1 - rmse_compare
        # Define min-max normalization function
        def min_max_transform(row):
            min_val = row.min()
            max_val = row.max()
            transformed_row = (row - min_val) / (max_val - min_val)
            return transformed_row

        # Apply min-max transform row-wise using apply and lambda function
        # then normalize so values in rows sum to 1
        wgt_df = wgt_df.apply(lambda row: min_max_transform(row), axis=1)
        wgt_df.columns = [i.replace('RMSE model #','') for i in wgt_df.columns]
        ensemble_df = pd.DataFrame()
        model_labels = [i.replace('RMSE model #', '') for i in rmse_compare.columns]
        for skill in rmse_compare.index:
            pred_values = pd.DataFrame(columns = df.columns)
            # gather the predicted values of each model
            skill_found = False
            for label in model_labels:
                if skill in dfs[label].index:
                    pred_values.loc[label, :] = dfs[label].loc[skill,:]
                    if not skill_found:
                        addl_values = dfs[label].loc[skill,[act_varname, 'Monthly average obs']]
                        skill_found = True
            if skill_found and wgt_df.loc[skill,:].sum() > 0:
                # merge in weights
                pred_values['weight'] = wgt_df.loc[skill,:].div(wgt_df.loc[skill,:].sum())
                pred_values['weight'] = pred_values['weight'].fillna(0)
                pred_values = pred_values.dropna()
                weighted_pred = (pred_values[pred_varname] * pred_values['weight']).sum()

                # measure agreement of models
                weighted_varname = pred_varname.replace('predicted','weighted predicted')
                pred_std = DescrStatsW(pred_values[pred_varname].dropna(), weights=pred_values['weight'].dropna()).std
                row = pd.Series([weighted_pred, pred_std], name = skill, index= [weighted_varname,'Prediction std dev'])

                # add values that are the same across all tools
                row = pd.concat([addl_values, row])
                row['Percentage Point change'] = row[weighted_varname] - row[act_varname]
                row['Percentage change'] = (row[weighted_varname] - row[act_varname]) / row[act_varname] * 100
                row['Number of Models'] = pred_values.loc[pred_values.weight > 0].shape[0]
                row['Average RMSE'] = rmse_compare.loc[skill,:].mean()
                ensemble_df = pd.concat([ensemble_df,row], axis = 1)

        ensemble_df = ensemble_df.T
        # add Model Variance based off of prediction standard deviation ratio to actual values
        ensemble_df['std_est_ratio'] = ensemble_df['Prediction std dev'] / ensemble_df[weighted_varname]

        # cut offs will be for low, medium, and high categories.
        ensemble_df.loc[(ensemble_df.std_est_ratio > .25),'Model Variance'] = 'High'
        ensemble_df.loc[(ensemble_df.std_est_ratio <= .25) & (ensemble_df.std_est_ratio > .1), 'Model Variance'] = 'Medium'
        ensemble_df.loc[(ensemble_df.std_est_ratio <= .1), 'Model Variance'] = 'Low'
        ensemble_df.loc[(ensemble_df['Number of Models'] == 1), 'Model Variance'] = 'N/A, only one model used'
        ensemble_df = ensemble_df.drop('std_est_ratio', axis=1)



    # remove predictions made on niche skills
    ensemble_df = ensemble_df.loc[ensemble_df['Monthly average obs'] >= min_monthly_obs]

    # merge in mean salary and most common occupation/industries.
    if output_occ_codes:
        occ_df = pd.read_csv('output/most common 3dig occupation codes for each ' + hierarchy_lvl + '.csv', index_col=0)
    else:
        occ_df = pd.read_csv('output/most common 3dig occupations for each '+hierarchy_lvl+'.csv', index_col=0)
    ind_df = pd.read_csv('output/most common 3dig industries for each '+hierarchy_lvl+'.csv', index_col=0)
    sal_df = pd.read_csv('output/'+hierarchy_lvl+'_salaries with means.csv', index_col=0)
    sal_df = sal_df.rename({'mean':'Mean Salary'}, axis=1)
    if hierarchy_lvl == 'skill':
        ensemble_df.index = [i.replace('Skill: ', '') for i in ensemble_df.index]
    elif hierarchy_lvl == 'subcategory':
        ensemble_df.index = [i.replace('Skill subcat: ', '') for i in ensemble_df.index]
    elif hierarchy_lvl == 'category':
        ensemble_df.index = [i.replace('Skill cat: ', '') for i in ensemble_df.index]

    ensemble_df = ensemble_df.merge(sal_df[['Mean Salary']], left_index=True, right_index = True)

    occ_df.columns = ['Most common occ','2nd most common occ', '3rd most common occ', '4th most common occ', '5th most common occ']
    ind_df.columns = ['Most common ind', '2nd most common ind', '3rd most common ind', '4th most common ind',
                      '5th most common ind']

    ensemble_df = ensemble_df.merge(occ_df, left_index=True, right_index = True)
    ensemble_df = ensemble_df.merge(ind_df, left_index=True, right_index=True)

    # reorder columns
    if model_selection == 'weighted average':
        # make sure we account for all columns in ensemble_df
        col_order = [act_varname,
           weighted_varname,
           'Percentage Point change', 'Percentage change', 'Model Variance',
           'Mean Salary', 'Most common occ', '2nd most common occ',
           '3rd most common occ', '4th most common occ', '5th most common occ',
           'Most common ind', '2nd most common ind', '3rd most common ind',
           '4th most common ind', '5th most common ind','Monthly average obs','Prediction std dev', 'Number of Models', 'Average RMSE']

        assert len([i for i in ensemble_df.columns if i not in col_order]) == 0, 'unexpected columns for ensemble_df'

        ensemble_df = ensemble_df[col_order]

    ensemble_df.to_csv('output/predicted changes/ensemble results '+title+'.csv')
    pass

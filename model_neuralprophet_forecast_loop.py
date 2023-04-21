'''
Luke Patterson
model_skforecast_forecast_loop.py

Purpose: define function to run Multilayer perceptron  model on each skill, generate forecasts, and log results
Input:
    COVID/chicago_covid_monthly.xlsx -> Covid hospitalization case counts for chicago
    One of:
        data/test monthly counts season-adj category.csv
        data/test monthly counts season-adj subcategory.csv
        data/test monthly counts season-adj skill.csv

Output:
        'result_logs/looped transformer model results '+ date_run+' '+run_name +'.csv' <- Log of parameters and performance metrics
        'output/predicted job posting shares '+date_run+' '+run_name+'.csv') <- Forecasted time series
'''

import pandas as pd
import numpy as np
import time
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from neuralprophet import NeuralProphet
import datetime
from utils import predQ, adf_test, invert_transformation, forecast_accuracy, visualize_predictions, results_analysis
from dateutil.relativedelta import relativedelta

def run_neuralprophet_loop(result_log = None, pred_df = None, start_val= 0, test_split = .1,
                         input_len_used = 4, period_past_data = None, targets_sample = None, min_month_avg = 50, min_tot_inc = 50
                         , ccc_taught_only = True, differenced = False, hierarchy_lvl = 'skill', past_months_data = None,
                         forecast_dampener = None,run_name = '', analyze_results = True, viz_sample = None):
    '''
    params:
        result_log - previous result log data frame
        pred_df - previous prediction results dataframe
        start_val - skill number to start at for interrupted runs
        input_len_used - how many months prior to train on
         period_past_data - how many time periods (months) worth of data to use. If None, use all data provided.
        targets_sample - length of subset of targets to train on; used for shortening runtimes of tests
        min_month_avg - minimum monthly average job postings for skill to be forecasted for
        min_tot_inc - minimum total increase between first and last observed month for skill to be forecasted for
        hierarchy_lvl - level of EMSI taxonomy to use: skill, subcategory, category
        param_grid_search - whether to perform a grid search for the best hyperparameters
        forecast_dampener -  forecasted changes are dampened by 1/(num of months out a forecast is) ^ forecast_dampener
        run_name - name to give run's log/results files.
        analyze_results - whether to run results analysis at the end of the run
        viz_sample - param to pass for results analysis

    Function to test run NeuralProphet model with various parameters, and understand runtime
    '''
    run_name = run_name + ' lvl ' + hierarchy_lvl
    date_run = datetime.datetime.now().strftime('%H_%M_%d_%m_%Y')

    if result_log is None:
        result_log = pd.DataFrame()

    assert (hierarchy_lvl in ['skill','subcategory', 'category'])
    df = pd.read_csv('data/test monthly counts season-adj '+hierarchy_lvl+'.csv', index_col=0)

    #--------------------
    # Feature Selection
    #-------------------

    if hierarchy_lvl == 'skill':
        # look only for those skills with mean 50 postings, or whose postings count have increased by 50 from the first to last month monitored

        raw_df = pd.read_csv('data/test monthly counts.csv')
        raw_df = raw_df.rename({'Unnamed: 0': 'date'}, axis=1)
        raw_df = raw_df.fillna(method='ffill')
        # 7-55 filter is to remove months with 0 obs
        raw_df = raw_df.iloc[7:55, :].reset_index(drop=True)
        # normalize all columns based on job postings counts
        raw_df = raw_df.drop('date', axis=1)

        # identify those skills who have from first to last month by at least 50 postings
        demand_diff = raw_df.T.iloc[:, -1] - raw_df.T.iloc[:, 0]
        targets = raw_df.mean(numeric_only=True).loc[(raw_df.mean(numeric_only=True)>min_month_avg)|(demand_diff > min_tot_inc)].index
    else:
        targets = [i for i in df.columns if 'Skill' in i]
    date_idx = pd.to_datetime(df.index)
    df = df.set_index(pd.DatetimeIndex(date_idx))

    # include on CCC-taught skills
    if hierarchy_lvl != 'skill' and ccc_taught_only:
        print('Warning: CCC taught only compatible with skill-level hierarchy')

    if hierarchy_lvl == 'skill' and ccc_taught_only:
        ccc_df = pd.read_excel('emsi_skills_api/course_skill_counts.xlsx')
        ccc_df.columns = ['skill', 'count']
        ccc_skills = ['Skill: ' + i for i in ccc_df['skill']]
        targets = set(ccc_skills).intersection(set(targets)).union(set(['Postings count']))
        targets = list(targets)
        targets.sort()

    #targets = df.columns[1:]
    targets = targets[start_val:]
    if targets_sample is not None:
        targets = targets[:targets_sample]

    # add in COVID case count data
    # covid_df = pd.read_csv('data/NYT COVID us-counties clean.csv')
    covid_df = pd.read_excel('COVID/chicago_covid_monthly.xlsx', index_col=0)
    covid_df.index = pd.to_datetime(covid_df.index)
    covid_df = covid_df.reset_index()
    covid_df['year'] = covid_df.year_month.apply(lambda x: x.year)
    covid_df['month'] = covid_df.year_month.apply(lambda x: x.month)
    covid_df = covid_df.rename({'icu_filled_covid_total': 'hospitalizations'}, axis=1)[
        ['year', 'month', 'hospitalizations']]

    # add 0 rows for pre-covid years
    for y in range(2018, 2020):
        for m in range(1, 13):
            covid_df = pd.concat(
                [covid_df, pd.DataFrame([[y, m, 0]], columns=['year', 'month', 'hospitalizations'])])
    covid_df = pd.concat([covid_df, pd.DataFrame([[2020, 1, 0]], columns=['year', 'month', 'hospitalizations'])])
    covid_df = pd.concat([covid_df, pd.DataFrame([[2020, 2, 0]], columns=['year', 'month', 'hospitalizations'])])

    # reshape to match the features data set and merge with features data
    covid_df = covid_df.sort_values(['year', 'month']).reset_index(drop=True)
    covid_df = covid_df.iloc[7:55, :]
    covid_df.index = date_idx
    covid_df = covid_df.drop(['year', 'month'], axis=1)

    df = df.merge(covid_df, left_index = True, right_index = True)

    orig_df = df.copy()

    # ------------------------
    # Model Execution
    #------------------------

    # set a variable to target
    print('Number of targets:',len(targets))
    if pred_df is None:
        pred_df = pd.DataFrame()
    features_main = df.corr()
    orig_df = df.copy()
    for n,t in enumerate(targets):
        df = orig_df
        # only forecast skills
        if 'Skill' not in t:
            continue
        # if no postings exist, skip skill
        if df[t].sum() == 0:
            continue
        start = datetime.datetime.now()
        print('Modeling',n,'of',len(targets),'skills')

        # option to perform differencing on non-stationary skills
        diffs_made = 0
        if differenced:
            # check to see if any of the series are non-stationary

            if df[t].sum() != 0:
                result = adf_test(df[t])
                if result > .05:
                    diffs_made += 1
                    df = df.diff().dropna()

                    # rerun stationary tests on results
                    result2 = adf_test(df[t])

                    # if still non-stationary, difference again
                    if result2 > .05:
                        diffs_made += 1
                        df = df.diff().dropna()


        # shorten df to only have time steps of length period_past_data
        # if not specified, use all data
        if period_past_data == None:
            period_past_data = df.shape[0]
        df = df.iloc[-period_past_data:]

        # figure out what features to use
        features = features_main[t]
        # filter to only those with at least a moderate correlation of .25
        features = features.loc[features.abs()> .25]
        features = features.drop(t).index

        # min max scale features
        X = df[features]
        X = pd.DataFrame(MinMaxScaler().fit_transform(X))
        X.columns = features
        X.index = df.index
        y = df[[t]].reset_index()
        #y = pd.DataFrame(MinMaxScaler().fit_transform(y), columns=['y'])
        y.columns = ['ds','y']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split, shuffle= False)
        if past_months_data is not None:
            y_train = y_train[-past_months_data:]
        forecaster = NeuralProphet(yearly_seasonality=False, weekly_seasonality=False,
                                   daily_seasonality=False,ar_reg=1, learning_rate=.01)
        metrics = forecaster.fit(y_train, freq="MS", progress="plot", epochs=200) #  validation_df=y_test['y'],
        fut_df = forecaster.make_future_dataframe(df=y_train, periods = 43)

        pred_row = forecaster.predict(fut_df)['yhat1']

        # option to dampen forecasts
        if forecast_dampener is not None:
            damp_df = pd.DataFrame()
            damp_df['date'] = fut_df['ds']
            damp_df['idx'] = np.arange(len(fut_df.index))
            damp_df['est'] = pred_row
            damp_df['est_diff'] = damp_df.est.diff(1)
            damp_df['est_shift'] = damp_df.est.shift(1)
            # forecasted changes are dampened by 1/(num of months out a forecast is) ^ forecast_dampener
            # this was implemented to dampen linear forecasts that moved outside of the historical domain of values when forecasting far ahead.
            damp_df['dampen_val'] = damp_df.loc[0, 'est'] + (damp_df.est_diff * 1/damp_df.idx ** forecast_dampener).cumsum()
            pred_row = damp_df['dampen_val']

        # make forecast date idx
        # min_date = datetime.date(2022, 3, 1)
        # max_date = min_date + relativedelta(months=+47)
        # dates = pd.period_range(min_date, max_date, freq='M')
        # fcast_date_idx = pd.DatetimeIndex(dates.to_timestamp())
        # pred_row.index = fcast_date_idx
        pred_row.index = fut_df['ds']
        pred_row.name = t

        # mark the test set for evaluation
        pred_short = pred_row[:len(y_test)]

        # revert differencing if any differences made
        if diffs_made > 0:
            if diffs_made == 2:
                pred_row = (df[t].iloc[-1] - df[t].iloc[-2]) + pred_row.cumsum()
            pred_row = df[t].iloc[-1] + pred_row.cumsum()
        # concatenate to df
        if pred_df.empty:
            pred_df = pd.DataFrame(index = pred_row.index)
        else:
            pred_df.index = pred_row.index
        pred_df = pd.concat([pred_df, pred_row],axis=1)

        # evaluate performance of the forecast
        # Use only the values with known values for assessing forecasting accuracy (the test data set)
        accuracy_prod = forecast_accuracy(pred_short, y_test.set_index('ds')['y'])

        row = pd.Series()
        row['target'] = t
        row['Normalized RMSE'] = accuracy_prod['rmse'] / (df[t].max() - df[t].min())
        row['MAPE'] = accuracy_prod['mape']
        row['runtime'] = datetime.datetime.now() - start
        row['num_features_used'] = len(X.columns)

        result_log = result_log.append(row, ignore_index=True)

        # log results
        result_log['timestamp'] = date_run
        # result_log['RMSE'] = perf_scores[0]
        # result_log['MAPE'] = perf_scores[1]
        result_log['num_features_raw'] = df.shape[1] - 2
        result_log['num_features_used'] = len(features)
        # result_log['pca_components'] = pca_components
        result_log['RUN_NAME'] = run_name
        result_log['ccc_taught_only'] = ccc_taught_only
        result_log['input_len_used'] = input_len_used

        pd.DataFrame(result_log).to_csv('result_logs/looped ProphetAR model results '+
                                          date_run+' '+run_name +
                                          '.csv')

        pred_df.to_csv('output/predicted job posting shares '+
                                          date_run+' '+run_name+
                                          '.csv')
    if analyze_results:
        print('visualizing results')
        visualize_predictions('predicted job posting shares ' + date_run + ' ' + run_name,
                              sample=viz_sample)
        results_analysis('predicted job posting shares ' + date_run + ' ' + run_name)

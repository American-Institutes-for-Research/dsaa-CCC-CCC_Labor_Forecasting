import os

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import sys
import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.stats.stattools import durbin_watson
from utils import grangers_causation_matrix, cointegration_test, adf_test, forecast_accuracy, invert_transformation_series, results_analysis, visualize_predictions


def run_ARIMA_loop(result_log = None, pred_df = None, start_val= 0, max_lags = 12, test_tvalues = 5,
                         input_len_used = 12, targets_sample = None, min_month_avg = 50, min_tot_inc = 50, cand_features_num=100
                         , ccc_taught_only = True, auto_reg = 3, max_diffs=2, moving_avg = 1, trend = None, use_exog = False,
                         hierarchy_lvl = 'skill', run_name = '', batch_name = None,
                         analyze_results = True, viz_predictions = True, viz_sample=None):
    '''
    params:
        result_log - previous result log data frame
        pred_df - previous prediction results dataframe
        start_val - skill number to start at for interrupted runs
        input_len_used - how many months prior to train on
        targets_sample - length of subset of targets to train on; used for shortening runtimes of tests
        min_month_avg - minimum monthly average job postings for skill to be forecasted for
        min_tot_inc - minimum total increase between first and last observed month for skill to be forecasted for
        auto_reg - autoregressive value of the ARIMA model
        max_diffs - maximum levels of differences to take when looking for a stationary series.
        moving_avg - moving average value of the ARIMA model
        trend - tuple representing the parameter controlling the deterministic trend of the ARIMA model.
        use_exog - whether to use exog variables in the ARIMA model
        hierarchy_lvl - level of EMSI taxonomy to use: skill, subcategory, category
        run_name - name to give run's log/results files.
        batch_name - folder with name of the batch of runs this run will be a part of
        analyze_results - whether to run results analysis at the end of the run
        viz_sample - param to pass for results analysis

    Function to test run transformer model with various parameters, and understand runtime

    Primarily based off of this post: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
    '''
    run_name = run_name + ' lvl ' + hierarchy_lvl

    # if using batch name, check if batch folders exist
    if batch_name is not None:
        batch_output = 'output/batch_'+batch_name
        batch_logs = 'result_logs/batch_'+batch_name
        if not os.path.exists(batch_logs):
            os.mkdir(batch_logs)
        if not os.path.exists(batch_output):
            os.mkdir(batch_output)
    else:
        batch_output = 'output/'
        batch_logs = 'result_logs/'
    date_run = datetime.datetime.now().strftime('%H_%M_%S_%d_%m_%Y')
    if result_log is None:
        result_log = pd.DataFrame()

    assert (hierarchy_lvl in ['skill', 'subcategory', 'category'])
    df = pd.read_csv('data/test monthly counts season-adj ' + hierarchy_lvl + '.csv', index_col=0)

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
        targets = set(ccc_skills).intersection(set(targets)).union({'Postings count'})
        targets = list(targets)
        targets.sort()

    # targets = df.columns[1:]
    targets = targets[start_val:]
    if targets_sample is not None:
        targets = targets[:targets_sample]

    # add in COVID case count data
    #covid_df = pd.read_csv('data/NYT COVID us-counties clean.csv')
    covid_df = pd.read_excel('COVID/chicago_covid_monthly.xlsx', index_col=0)
    covid_df.index = pd.to_datetime(covid_df.index)
    covid_df = covid_df.reset_index()
    covid_df['year'] = covid_df.year_month.apply(lambda x: x.year)
    covid_df['month'] = covid_df.year_month.apply(lambda x: x.month)
    covid_df = covid_df.rename({'icu_filled_covid_total':'hospitalizations'},axis=1)[['year','month','hospitalizations']]

    # add 0 rows for pre-covid years
    for y in range(2018,2020):
        for m in range(1,13):
            covid_df = pd.concat([covid_df, pd.DataFrame([[y, m, 0]], columns= ['year','month','hospitalizations'])])
    covid_df = pd.concat([covid_df, pd.DataFrame([[2020, 1, 0]], columns=['year', 'month', 'hospitalizations'])])
    covid_df = pd.concat([covid_df, pd.DataFrame([[2020, 2, 0]], columns=['year', 'month', 'hospitalizations'])])

    # reshape to match the features data set and merge with features data
    covid_df = covid_df.sort_values(['year','month']).reset_index(drop=True)
    covid_df = covid_df.iloc[7:55,:]
    covid_df.index = date_idx
    covid_df = covid_df.drop(['year','month'],axis=1)

    df = df.merge(covid_df, left_index = True, right_index = True)


    # ------------------------
    # Model Execution
    #------------------------

    # set a variable to target
    print('Number of targets:',len(targets))
    if pred_df is None:
        pred_df = pd.DataFrame()
    features_main = df.corr()
    target_tracker = pd.Series(index=targets)
    for n,t in enumerate(targets):
        # only perform for skill variables
        if 'Skill' not in t:
            continue

        # skip if no observations of skill exist
        if df[t].sum() == 0:
            continue
        start = datetime.datetime.now()
        print('Modeling',n,'of',len(targets),'skills')

        # loop to try testing multiple sets of variables until we find one that's causally correlated with the target
        success = False
        count = 0

        # figure out what features to use - try just top cand_features_num most correlated features that are not the target
        features = features_main[t].abs().sort_values(ascending=False).dropna()
        if cand_features_num > len(features):
            cand_features_num = len(features)
        features = features[:cand_features_num+1].drop(t)


        df_feat = df[list(features.index) + [t]]

        # issues with duplicate columns at this point
        df_feat = df_feat.T.drop_duplicates().T

        # create training and test data
        df_train, df_test = df_feat[0:-test_tvalues], df_feat[-test_tvalues:]

        # check to see if any of the series are non-stationary
        diffs_made = 0

        # for c in df_feat.columns:
        #     if df_feat[c].sum() != 0:
        #         result = adf_test(df_feat[c])
        #         # if result is not significant, series is non-stationary
        #         if result > .05:
        #             diff_tracker[c] = 'Non-stationary'
        #         else:
        #             diff_tracker[c] = 'Stationary'

        # check to see target column is stationary
        result = adf_test(df_train[t])
        stationary = result <= .05


        # if any of the series are non-stationary, difference the results. repeat until all stationary or diffs_made is > 2
        # not sure what upper limit of differences should be, but literature seems to suggest at most 2 should be needed,
        # and too many more might dilute predictive power
        df_differenced = df_train
        while not stationary and diffs_made < max_diffs:
            diffs_made += 1
            df_differenced = df_differenced.diff().dropna()
            # rerun stationary tests on results
            # for c in df_differenced.columns:
            #     result = adf_test(df_differenced[c])
            #     # if result is not significant, series is non-stationary
            #     if result > .05:
            #         diff_tracker[c] = 'Non-stationary'
            #     else:
            #         diff_tracker[c] = 'Stationary'
            result = adf_test(df_differenced[t])
            stationary = result <= .05

        assert(diffs_made<=max_diffs)

        # note whether non-stationarity still existed after differencing
        if not stationary:
            print('Warning - non-stationary values still detected after',max_diffs, 'difference levels')
            target_tracker[t] = 'Warning - non-stationary values still detected after differencing'


        if use_exog:
            model = ARIMA(endog = df_differenced[t], exog=df_differenced.drop(t, axis=1), order= (auto_reg,diffs_made,moving_avg), freq='MS', trend = trend)
        else:
            model = ARIMA(endog=df_differenced[t], order= (auto_reg,diffs_made,moving_avg), freq='MS', trend = trend)
        model.initialize_approximate_diffuse()
        model_fitted = model.fit()

        if use_exog:
            pred_row = model_fitted.forecast(exog=df_differenced.drop(t, axis=1), steps = 48 - test_tvalues)
        else:
            pred_row = model_fitted.forecast(steps=48 - test_tvalues)
        pred_row.name = t

        # print model summary to log file
        old_stdout = sys.stdout
        log_file = open("result_logs/ARIMA_models/ARIMA model summary "+t.replace(":", "").replace('/','').replace("*",'')+".txt", "w")
        sys.stdout = log_file
        try:
            print(model_fitted.summary())
        except Exception as e:
            print(e)
        sys.stdout = old_stdout
        log_file.close()

        # conduct forecast
        forecast_input = df_differenced


        # create forecasted date index; our forecast starts at the first time step of the test data set and extends
        # 43 - input_len_used time steps forward
        min_date = datetime.date(2022, 3, 1)
        max_date = min_date + relativedelta(months=+42-input_len_used)
        dates = pd.period_range(min_date, max_date, freq='M')
        date_idx = pd.DatetimeIndex(dates.to_timestamp())

        if diffs_made > 0:
            pred_row = invert_transformation_series(df_train, pred_row, diffs_made)

        # add results
        # convert to dataframe

        pred_row.name = t

        # concatenate to df
        if pred_df.empty:
            pred_df = pd.DataFrame(index=pred_row.index)
        else:
            pred_df.index = pred_row.index
        pred_df = pd.concat([pred_df, pred_row], axis=1)
        pred_df.to_csv(batch_output+'/predicted job posting shares ' +
                       date_run + ' ' + run_name +
                       '.csv')

        # evaluate performance of the forecast


        # Use only the values with known values for assessing forecasting accuracy (the test data set)
        pred_row_short = pred_row.iloc[:test_tvalues]
        accuracy_prod = forecast_accuracy(pred_row_short, df_test[t])


        # record results
        row = pd.Series()
        row['target'] = t
        row['model training result'] = target_tracker.loc[t]
        row['runtime'] = datetime.datetime.now() - start
        row['num_features_used'] = len(df_feat.columns)
        row['Normalized RMSE'] = accuracy_prod['rmse']/(df[t].max() - df[t].min())
        row['MAPE']= accuracy_prod['mape']

        result_log = result_log.append(row, ignore_index=True)

        # log results
        result_log['timestamp'] = date_run
        result_log['num_features_raw'] = df.shape[1] - 2
        result_log['RUN_NAME'] = run_name
        result_log['ccc_taught_only'] = ccc_taught_only
        result_log['input_len_used'] = input_len_used
        result_log['differences_made'] = diffs_made
        result_log['cand_features_num'] = cand_features_num
        result_log['auto_reg'] = auto_reg
        result_log['moving_avg'] = moving_avg
        result_log['trend'] = str(trend)
        result_log['use_exog'] = use_exog

        pd.DataFrame(result_log).to_csv(batch_logs+'/looped ARIMA model results '+
                                          date_run+' '+run_name +
                                          '.csv')

    target_tracker.to_csv(batch_logs+'/looped ARIMA model variable training success tracker ' +
                                      date_run + ' ' + run_name +
                                      '.csv')


    if analyze_results:
        print('analyzing results')
        results_analysis('predicted job posting shares ' + date_run + ' ' + run_name)

    if viz_predictions:
        print('visualizing results')
        visualize_predictions('predicted job posting shares ' + date_run + ' ' + run_name,
                              sample=viz_sample)

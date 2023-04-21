'''
Luke Patterson
DLPM_forecast_loop.py

Purpose: define function to run dynamic linear panel model on each skill, generate forecasts, and log results
Input:
    COVID/chicago_covid_monthly.xlsx -> Covid case counts for chicago
    One of:
        data/test monthly counts season-adj category.csv
        data/test monthly counts season-adj subcategory.csv
        data/test monthly counts season-adj skill.csv

Output:
        'result_logs/looped panel model results '+ date_run+' '+run_name +'.csv' <- Log of parameters and performance metrics
        'output/predicted job posting shares '+date_run+' '+run_name+'.csv') <- Forecasted time series
'''

from datetime import datetime, timedelta
import pandas as pd
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts import TimeSeries
#from linearmodels import PanelOLS
import numpy as np
from utils import forecast_accuracy, visualize_predictions, results_analysis
# from dateutil.relativedelta import relativedelta

def run_DLPM_loop(result_log = None, pred_df = None, start_val= 0, test_tvalues =5,
                         input_len_used = 12, targets_sample = None, min_month_avg = 50, min_tot_inc = 50, cand_features_num=20
                         , output_chunk_length = None, ccc_taught_only = True, pred_length = None, hierarchy_lvl = 'skill', run_name = '', visualize_results = True,
                  viz_sample = None):
    '''
    params:
        result_log - previous result log data frame
        pred_df - previous prediction results dataframe
        start_val - skill number to start at for interrupted runs
        test_tvalues - number of time steps to put in test data set
        input_len_used - how many months prior to train on
        output_chunk_length - Number of time steps predicted at once by the internal regression model.
        cand_features_num - number of feature to select
        targets_sample - length of subset of targets to train on; used for shortening runtimes of tests
        min_month_avg - minimum monthly average job postings for skill to be forecasted for
        min_tot_inc - minimum total increase between first and last observed month for skill to be forecasted for
        hierarchy_lvl - level of EMSI taxonomy to use: skill, subcategory, category
        run_name - name to give run's log/results files.

    Function to test run DLPM model with various parameters, and understand runtime
    '''
    run_name = run_name + ' lvl ' + hierarchy_lvl
    date_run = datetime.now().strftime('%H_%M_%d_%m_%Y')
    if result_log is None:
        result_log = pd.DataFrame()

    assert (hierarchy_lvl in ['skill', 'subcategory', 'category'])
    df = pd.read_csv('data/test monthly counts season-adj county ' + hierarchy_lvl + '.csv', index_col=0)
    #df = pd.read_csv('data/test monthly counts county panel season-adj.csv', index_col=0)
    #--------------------
    # Feature Selection
    #-------------------

    # look only for those skills with mean 50 postings, or whose postings count have increased by 50 from the first to last month monitored

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
        targets = raw_df.mean(numeric_only=True).loc[
            (raw_df.mean(numeric_only=True) > min_month_avg) | (demand_diff > min_tot_inc)].index
    else:
        targets = [i for i in df.columns if 'Skill' in i]

    date_idx = pd.DatetimeIndex(df.date)
    county_idx = df.index
    df = df.set_index([county_idx, date_idx])

    df = df[~df.index.duplicated(keep='first')]

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

    targets = targets[start_val:]
    if targets_sample is not None:
        targets = targets[:targets_sample]
    targets = [i for i in targets if 'Skill' in i]

    # add in COVID case count data
    # covid_df = pd.read_csv('data/NYT COVID us-counties clean.csv')
    # # add 0 rows for pre-covid years
    # for y in range(2018,2020):
    #     for m in range(1,13):
    #         covid_df = covid_df.append(pd.Series([y, m, 0], index= ['year','month','cases_change']), ignore_index = True)
    #
    # # reshape to match the features data set and merge with features data
    # covid_df = covid_df.sort_values(['year','month'])
    # covid_df = covid_df.iloc[7:,:]
    # covid_df.index = date_idx
    # covid_df = covid_df.drop(['year','month'],axis=1)
    # covid_df.columns = ['covid_cases']
    # targets = targets.union(['covid_cases'])

    # df = df.merge(covid_df, left_index = True, right_index = True)

    # --------------------------
    # Model Execution
    # --------------------------
    print('Number of targets:', len(targets))
    if pred_df is None:
        pred_df = pd.DataFrame()
    # TODO: implement additional covariates beyond just target skills
    features_main = df[targets].corr()

    for n, t in enumerate(targets):
        start = datetime.now()
        print('Modeling', n, 'of', len(targets), 'skills')
        endog= df[t]

        # test using most correlated features
        features = features_main[t].abs().sort_values(ascending=False).dropna()
        features = features.drop(t).iloc[:cand_features_num + 1]

        exogs = df[list(features.index) + [t]]
        min_t = min(endog.index.get_level_values(1))  # starting period
        T = len(date_idx.drop_duplicates())  # total number of periods

        # Names for the original and differenced endog and exog vars
        ename = endog.name
        Dename = 'D' + ename
        xnames = exogs.columns.tolist()
        Dxnames = ['D' + x for x in xnames]

        # We'll store all of the data in a dataframe
        data = pd.DataFrame()
        data[ename] = endog
        data[Dename] = endog.groupby(level=0).diff()

        # Generate and store the lags of the differenced endog variable
        Lenames = []
        LDenames = []
        for k in range(1, input_len_used + 1):
            col = 'L%s%s' % (k, ename)
            colD = 'L%s%s' % (k, Dename)
            Lenames.append(col)
            LDenames.append(colD)
            data[col] = data[ename].shift(k)
            data[colD] = data[Dename].shift(k)

        # Store the original and the diffs of the exog variables
        for x in xnames:
            data[x] = exogs[x]
            data['D' + x] = exogs[x].groupby(level=0).diff()

        # commented out, trying a non-IV model instead
        # # Set up the instruments -- lags of the endog levels for different time periods
        # instrnames = []
        # for n, t in enumerate(date_idx.drop_duplicates()):
        #     for k in range(1, input_len_used + 1):
        #         col = 'ILVL_t%iL%i' % (n, k)
        #         instrnames.append(col)
        #         data[col] = endog.groupby(level=0).shift(k)
        #         data.loc[endog.index.get_level_values(1) != t, col] = 0
        #
        # dropped = data.dropna()
        # dropped['CLUSTER_VAR'] = dropped.index.get_level_values(0)
        #
        # # make sure columns are not duplicated to ensure full rank
        # zero_cols = dropped.sum().loc[dropped.sum()==0].index
        # instrnames = [i for i in instrnames if i not in zero_cols]
        # dupes = dropped[instrnames].sum().duplicated()
        # instrnames = [i for i in instrnames if i not in dupes.loc[dupes].index and 'L1' in i]
        # model = IVGMM(dropped[Dename], dropped[Dxnames], dropped[LDenames], dropped[instrnames].iloc[:,:12],
        #                    weight_type='clustered', clusters=dropped['CLUSTER_VAR'])
        # model.fit()


        # detect linearly dependent row vectors
        #dropped = data.dropna()
        data = data.reset_index().rename({'level_0': 'county', 'level_1': 'date'}, axis=1)
        data = data.set_index('date')

        # exogs = {}
        train_targets = {}
        test_targets = {}
        train_exogs = {}
        test_exogs = {}
        counties = data.county.unique()
        for c in counties:
            train_data = data.loc[data.county == c].iloc[0:-test_tvalues,:]
            test_data = data.loc[data.county == c]
            train_targets[c] = TimeSeries.from_series(train_data[t].copy(), fill_missing_dates=True, freq='MS')
            test_targets[c] = TimeSeries.from_series(test_data[t].copy(), fill_missing_dates=True, freq='MS')
            train_exogs[c] = TimeSeries.from_series(train_data[xnames].copy(), fill_missing_dates=True, freq='MS')
            test_exogs[c] = TimeSeries.from_series(test_data[xnames].copy(), fill_missing_dates=True, freq='MS')
        # model = PanelOLS(dependent=train_data[Dename], exog = train_data[Dxnames], entity_effects= True, time_effects= True,
        #                  check_rank= False, drop_absorbed=True)
        # fitted = model.fit()
        # model = LinearRegressionModel(output_chunk_length=18, lags_past_covariates=12)
        if output_chunk_length is None:
            output_chunk_length = 43 - input_len_used

        model = LinearRegressionModel(output_chunk_length=output_chunk_length, lags_past_covariates=input_len_used)
        model.fit(list(train_targets.values()), past_covariates=list(train_exogs.values()))

        if pred_length is None:
            pred_length = 43-input_len_used
        ts_tpreds_long = model.predict(n=pred_length, series=list(train_targets.values()), past_covariates=list(train_exogs.values()))
        # model = LinearRegressionModel(output_chunk_length=43 - input_len_used, lags = input_len_used)
        # model.fit(list(train_targets.values()))
        # ts_tpreds_long = model.predict(n=43, series=list(test_targets.values()))

        pred_row = pd.Series()
        for n, ts_tpred_long in enumerate(ts_tpreds_long):
            # mark the test set for evaluation
            ts_tpred = ts_tpred_long[:test_tvalues]

            # take the rest of the predictions and transform them back into a dataframe
            ts_tfut = ts_tpred_long
            idx = pd.MultiIndex.from_arrays(
                [np.array([counties[n] for _ in range(len(ts_tfut.time_index.values))]),
                 ts_tfut.time_index.date.astype(str)])

            # first put all county predictions into a single series
            if pred_row is None:
                pred_row = pd.Series([i[0] for i in ts_tfut.values()], index = idx, name =t)
            else:
                pred_row = pd.concat([pred_row,pd.Series([i[0] for i in ts_tfut.values()], index = idx, name =t)], axis=0)
        pred_row.name = t
        pred_row.index = pd.MultiIndex.from_tuples(pred_row.index)
        # add series to dataframe
        if pred_df.empty:
            pred_df = pd.DataFrame(index=pred_row.index)
        pred_df = pd.concat([pred_df, pred_row], axis=1)
        pred_df.index = pd.MultiIndex.from_tuples(pred_df.index)

        # calculate RMSE and MAPE for each county
        for c in counties:
            cdf = df.loc[c, t]
            cpred_df = pred_df.loc[c, t]

            accuracy_prod = forecast_accuracy(cpred_df.iloc[0:test_tvalues].values, cdf[-test_tvalues:].values)

            row = pd.Series()
            row['target'] = t
            row['county'] = c
            row['Normalized RMSE'] = accuracy_prod['rmse'] / (cdf.max() - cdf.min())
            row['MAPE'] = accuracy_prod['mape']
            row['runtime'] = datetime.now() - start
            row['num_features_used'] = len(exogs)

            result_log = result_log.append(row, ignore_index=True)

            # log results
            result_log['timestamp'] = date_run
            result_log['num_features_raw'] = df.shape[1] - 2
            result_log['RUN_NAME'] = run_name
            result_log['ccc_taught_only'] = ccc_taught_only
            result_log['input_len_used'] = input_len_used

        pd.DataFrame(result_log).to_csv('result_logs/looped panel model results '+
                                          date_run+' '+run_name +
                                          '.csv')

        pred_df.to_csv('output/predicted job posting shares '+
                                          date_run+' '+run_name+
                                          '.csv')
    if visualize_results:
        print('visualizing results')
        visualize_predictions('predicted job posting shares '+date_run+' '+run_name, panel_data=True, sample = viz_sample)
        results_analysis('predicted job posting shares '+date_run+' '+run_name, panel_data=True)

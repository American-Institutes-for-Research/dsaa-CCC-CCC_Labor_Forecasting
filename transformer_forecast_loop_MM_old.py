'''
Luke Patterson & Mason Miller (edits & additions)
transformer_forecast_loop.py

Purpose: define function to run ML transformer model on each skill, generate forecasts, and log results
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

# adapting methods from
# https://towardsdatascience.com/transformer-unleashed-deep-forecasting-of-multivariate-time-series-in-python-9ca729dac019
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from datetime import datetime
import torch

from utils import predQ, adf_test, invert_transformation, visualize_predictions, results_analysis#, drop_zero_posting_months

def run_transformer_loop_MM(EPOCHS=20,N_SAMPLES = 100,DIM_FF = 128,HEADS = 4,ENCODE = 4, DECODE = 4,
                            BATCH = 32, SPLIT=.75, result_log = None, pred_df = None, start_val=None, end_val=None,
                         input_len_used = 12, corr_filter=.95, period_past_data = None,
                            min_month_avg = 50, min_tot_inc = 50,ccc_taught_only = True, differenced = False,
                            hierarchy_lvl = 'skill',output_chunk_len = None, run_name = '',
                         analyze_results = True, viz_sample=None, exit_loop=False,
                            features_print=False, features_export=False):
    '''
    params:
       EPOCHS - number of epocs the model trains on
       N_SAMPLES - number of times a prediction is sampled from a probabilistic model
       DIM_FF - dimensions of the feedforward network
        HEADS -  The number of heads in the multi-head attention mechanism
        ENCODE - encoder layers
        DECODE - decoder layers
        BATCH - batch size
        SPLIT - Train/test split %
        LEARN - learning rate; default 1e-3
        lr_scheduler_cls - learning rate scheduler (reduces learning rate by EPOCH to prevent overfitting); default=None
        result_log - previous result log data frame
        pred_df - previous prediction results dataframe
        start_val - skill number to start at for interrupted runs
        end_val - skill number to end at for interrupted runs or runs on a single skill only
        corr_filter - minimum pearson correlation value btwn target for skill to be included as a feature
        input_len_used - how many months prior to train on
        period_past_data - how many time periods (months) worth of data to use. If None, use all data provided.
        min_month_avg - minimum monthly average job postings for skill to be forecasted for
        min_tot_inc - minimum total increase between first and last observed month for skill to be forecasted for
        hierarchy_lvl - level of EMSI taxonomy to use: skill, subcategory, category
        pred_length - how many months out to make predictions for
        run_name - name to give run's log/results files.
        analyze_results - whether to run results analysis at the end of the run
        viz_sample - param to pass for results analysis
        features_report - exports file listing features of any prediction target
        exit_loop - breaks function prior to model fit, instead returning model inputs for visual inspection (default False)
        features print - prints # of features 1) passing correlation filter, 2) surviving minmax rescaler

    Function to test run transformer model with various parameters, and understand runtime
    '''
    run_name = run_name + ' lvl ' + hierarchy_lvl
    date_run = datetime.now().strftime('%H_%M_%d_%m_%Y')
    FEAT = 32           # d_model = number of expected features in the inputs, up to 512

    ACTF = "relu"       # activation function, relu (default) or gelu
    LEARN = 1e-3        # learning rate
    VALWAIT = 1         # epochs to wait before evaluating the loss on the test/validation set
    DROPOUT = 0.1       # dropout rate

    RAND = 42           # random seed
    N_JOBS = 3          # parallel processors to use;  -1 = all processors

    # default quantiles for QuantileRegression
    QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]


    FIGSIZE = (9, 6)

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
        #raw_df = drop_zero_posting_months(raw_df)
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

    if start_val is not None:
        if end_val is not None:
            targets = targets[start_val:end_val]
        else:
            targets = targets[start_val:]

    # add in COVID case count data
    covid_df = pd.read_excel('COVID/chicago_covid_monthly.xlsx', index_col=0)
    covid_df.index = pd.to_datetime(covid_df.index)
    covid_df = covid_df.reset_index()
    covid_df['year'] = covid_df.year_month.apply(lambda x: x.year)
    covid_df['month'] = covid_df.year_month.apply(lambda x: x.month)
    covid_df = covid_df.rename(columns={'acute_used_all_covid':'acute_filled_covid_total'})
    covid_df = covid_df[['year', 'month', 'icu_filled_covid_total', 'acute_filled_covid_total',
    'icu_surge_adult','icu_surge_pediatric','vent_surge_capacity']]
    covid_cols = covid_df.columns.tolist()

    # add 0 rows for pre-covid years
    row = {}
    for col in covid_cols:
        row[col] = 0
    for y in range(2018, 2020): # all months for 2018 and 2019
        for m in range(1, 13):
            row['year'] = y
            row['month'] = m
            covid_df = covid_df.append(row, ignore_index=True)
    for m in range(1,3): # 1/2020 and 2/2020
        row['year'] = 2020
        row['month'] = m
        covid_df = covid_df.append(row, ignore_index=True)

    # reshape to match the features data set and merge with features data
    covid_df = covid_df.sort_values(['year', 'month']).reset_index(drop=True)
    covid_df.index = pd.to_datetime(covid_df['month'].astype(str) + '/' + covid_df['year'].astype(str))
    covid_df = covid_df.drop(['year', 'month'], axis=1)

    orig_df = df.copy()

    # ------------------------
    # Model Execution
    #------------------------

    # set a variable to target
    print('Number of targets:',len(targets))
    if pred_df is None:
        pred_df = pd.DataFrame()
    for n,t in enumerate(targets):
        df = orig_df
        # only forecast skills
        if 'Skill' not in t:
            continue
        # if no postings exist, skip skill
        if df[t].sum() == 0:
            continue
        start = datetime.now()
        print('Modeling',n,'of',len(targets),'skills')
        skill = t
        skill = skill.replace("Skill: ", '')

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

        # figure out what features to use
        features = pd.DataFrame(df.corrwith(df[t]), columns=['corr'])
        # filter to only those with at least a moderate correlation of .25
        features = features[abs(features['corr'])>=corr_filter]
        features = features.sort_values(by='corr', ascending=False)

        if features_export:
            features.to_csv("output/Mason/" + skill + "_skill_correlations.csv")
            print('skill correlations exported')

        # prepare features for splitting and scaling (includes adding COVID features)
        feat_cols = features.index.tolist()
        df.index = pd.to_datetime(df.index)
        df = df[feat_cols]
        tf = df.merge(covid_df, how='left', left_index=True, right_index=True) # tf = target & features
        tf['date'] = tf.index

        # Convert target and features to darts TimeSeries objects
        # target
        target = 'Skill: Equipment Calibration'
        tts = TimeSeries.from_dataframe(tf[[target, 'date']], fill_missing_dates=True, freq=None, time_col='date')
        # features
        skill_feature_cols = [x for x in tf.columns.tolist() if x is not target]
        skill_features = tf[skill_feature_cols]
        fts = TimeSeries.from_dataframe(skill_features, fill_missing_dates=True, freq=None, time_col='date')

        # Train/validation split
        target_train, target_val = tts.split_after(SPLIT)
        features_train, features_val = fts.split_after(SPLIT)

        # MinMax scale target and features
        minmax = MinMaxScaler(feature_range=(0, 1))
        target_scaler = Scaler(minmax)
        target_train_scaled = target_scaler.fit_transform(target_train)
        target_val_scaled = target_scaler.transform(target_val)
        feature_scaler = Scaler(minmax)
        features_train_scaled = feature_scaler.fit_transform(features_train)
        features_val_scaled = feature_scaler.transform(features_val)

        # report on how many features survive correlation filter
        if features_print:
            print(skill, 'has', len(feat_cols), 'skills with >=', corr_filter, 'absolute value pearson correlation')

        # check whether all features survived minmax scaling
        assertion_msg = 'mismatch between number of features before and after scaling'
        #print(len(tf.columns.tolist()))
        #print(len(features_train_scaled.columns))
        assert len(tf.columns.tolist())-1 == len(features_train_scaled.columns), assertion_msg

        if exit_loop:
            print(skill)
            print('model inputs exported; now exiting function')
            return None

        # Setting TransformerModel() parameters
        input_len = 6
        output_len = 3

        count = 0
        while True:
            try:
                model = TransformerModel(
                    input_chunk_length=input_len,
                    output_chunk_length=output_len,
                    batch_size=BATCH,
                    n_epochs=EPOCHS,
                    model_name="Transformer_MM",
                    nr_epochs_val_period=VALWAIT,
                    d_model=64,
                    nhead=HEADS,
                    num_encoder_layers=ENCODE,
                    num_decoder_layers=DECODE,
                    dim_feedforward=DIM_FF,
                    dropout=DROPOUT,
                    activation=ACTF,
                    random_state=RAND,
                    likelihood=QuantileRegression(quantiles=QUANTILES),
                    #optimizer_cls={torch.optim.Adam},
                    #optimizer_kwargs={'lr':0.001},
                    #lr_scheduler_cls={torch.optim.lr_scheduler.ReduceLROnPlateau},
                    save_checkpoints=True,
                    force_reset=True
                )
                model.fit(series=target_train_scaled,
                          val_series=target_val_scaled,
                          past_covariates=features_train_scaled,
                          val_past_covariates=features_val_scaled,
                                verbose=True)
                print('model successfully fit')
                break
            except (FileNotFoundError, PermissionError, FileExistsError):
                if count < 20:
                    print('PermissionError, retrying')
                    import time
                    time.sleep(1)
                    count += 1
                    continue
                else:
                    raise('too many attempts at model training')

        # mark the test set for evaluation
        prediction_window = 48
        ts_tpred_long = model.predict(n=prediction_window,
                                    series = target_train_scaled,
                                    val_series=target_val_scaled,
                                    past_covariates = features_train_scaled,
                                    val_past_covariates=features_val_scaled,
                                    num_samples=N_SAMPLES,
                                    n_jobs=N_JOBS,
                                    verbose=True)
        print('completed predictions')

        # take the rest of the predictions and transform them back into a dataframe
        ts_tfut = ts_tpred_long
        ts_tpred = ts_tpred_long[:len(ts_ttest)]
        # remove the scaler transform
        ts_tfut = scalerP.inverse_transform(ts_tfut)

        # convert to dataframe

        pred_row = ts_tfut.quantile_df()
        pred_row = pred_row.iloc[:,0].apply(lambda x: float(x.values))
        pred_row.name = pred_row.name.replace('_0.5','')

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

        # retrieve forecast series for chosen quantiles,
        # inverse-transform each series,
        # insert them as columns in a new dataframe dfY
        q50_RMSE = np.inf
        q50_MAPE = np.inf
        ts_q50 = None
        pd.options.display.float_format = '{:,.2f}'.format
        dfY = pd.DataFrame()
        dfY["Actual"] = TimeSeries.pd_series(ts_test)

        # call helper function predQ, once for every quantile
        perf_scores = [predQ(ts_tpred, q, scalerP, ts_test) for q in QUANTILES]
        perf_scores = perf_scores[3]
        row = pd.Series()
        row['target'] = t
        row['Normalized RMSE'] = perf_scores[0]/(df[t].max() - df[t].min())
        row['MAPE']= perf_scores[1]
        row['runtime'] = datetime.now() - start
        row['num_features_used'] = len(df_feat.columns)

        result_log = result_log.append(row, ignore_index=True)

        # log results
        result_log['timestamp'] = date_run
        # result_log['RMSE'] = perf_scores[0]
        # result_log['MAPE'] = perf_scores[1]
        result_log['num_features_raw'] = df.shape[1] - 2
        result_log['num_features_used'] = len(features)
        # result_log['pca_components'] = pca_components
        result_log['EPOCHS'] = EPOCHS
        result_log['DIM_FF'] = DIM_FF
        result_log['N_SAMPLES'] = N_SAMPLES
        result_log['HEADS'] = HEADS
        result_log['ENCODE'] = ENCODE
        result_log['DECODE'] = DECODE
        result_log['BATCH'] = BATCH
        result_log['RUN_NAME'] = run_name
        result_log['ccc_taught_only'] = ccc_taught_only
        result_log['input_len_used'] = input_len_used

        pd.DataFrame(result_log).to_csv('result_logs/looped transformer model results '+
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

    pass

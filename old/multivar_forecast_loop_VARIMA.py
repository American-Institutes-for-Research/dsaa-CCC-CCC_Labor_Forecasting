# adapting methods from
# https://towardsdatascience.com/transformer-unleashed-deep-forecasting-of-multivariate-time-series-in-python-9ca729dac019
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.varima import VARIMA
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from datetime import datetime

def prepare_data():
    '''
    load data in preparation for running the transformer loop over each feature
    '''

    # df = pd.read_csv('data/test monthly counts 09302022.csv')
    df = pd.read_csv('../data/test monthly counts.csv')
    df = df.rename({'Unnamed: 0':'date'}, axis=1)
    df['month']= df['date'].str[5:7].astype('int')
    df = df.fillna(method='ffill')
    # 7-55 filter is to remove months with 0 obs
    df = df.iloc[7:55,:].reset_index(drop=True)

    # create times series index
    date_idx = pd.to_datetime(df['date'])

    # normalize all columns based on job postings counts
    df = df.drop('date', axis=1)
    job_counts = df['Postings count'].copy()
    raw_df = df.copy()
    df = df.divide(job_counts, axis=0)
    df['Postings count'] = job_counts
    df = df.set_index(pd.DatetimeIndex(date_idx))

    # establish target columns as ones with an average obs count over 100
    targets = raw_df.mean(numeric_only=True).loc[raw_df.mean(numeric_only=True)>100].index

    # filter to only skills trained by CCC
    ccc_df = pd.read_excel('emsi_skills_api/course_skill_counts.xlsx')
    ccc_df.columns = ['skill', 'count']
    ccc_skills = ['Skill: ' + i for i in ccc_df['skill']]
    targets = set(ccc_skills).intersection(set(targets)).union(set(['Postings count']))
    return df, targets

def run_VARIMA_loop(df, targets, result_log = None, pred_df = None, start_val= 0):
    '''
    params:
        df - job posting counts dataframe
        targets - set of targets to run loop on
        result_log - previous result log data frame
        pred_df - previous prediction results dataframe
        start_val - skill number to start at for interrupted runs
    Function to test run VARIMA model with various parameters, and understand runtime
    Just runs on a single target
    '''
    SPLIT = .9
    date_run = datetime.now().strftime('%H_%M_%d_%m_%Y')

    if result_log is None:
        result_log = pd.DataFrame()
    targets = list(targets)

    targets = targets[start_val:]

    # set a variable to target
    print('Number of targets:',len(targets))
    if pred_df is None:
        pred_df = pd.DataFrame()
    for n,t in enumerate(targets):
        start = datetime.now()
        print('Modeling',n,'of',len(targets),'skills')

        # figure out what features to use
        features = df.corr()[t]
        # filter to only those with at least a moderate correlation of .25
        features = features.loc[features.abs()> .25]
        features = features.drop(t).index

        # min max scale features
        df_feat = df[features]
        df_feat = pd.DataFrame(MinMaxScaler().fit_transform(df_feat))

        # # run PCA to reduce number of features
        # pca = PCA(n_components=min(pca_components, len(features)))
        # res_pca = pca.fit_transform(df_feat)
        #
        # # collect principal components in a dataframe
        # df_pca = pd.DataFrame(res_pca)
        # df_pca.index = df.index
        # df_pca = df_pca.add_prefix("pca")
        # df_pca[t] = df[t]

        # select pcas with correlation >.10
        # removing this part as it was just done for ease of explanability
        # selected_pca = df_pca.corr()[t].loc[df_pca.corr()[t].abs() > .1].drop(t).index

        #df.corr().to_csv('data/test corr.csv')

        # convert target to time series
        ts_P = TimeSeries.from_series(df[t], fill_missing_dates=True, freq=None)
        #ts_P = pd.Series([i[0] for i in ts_P.values()])
        # convert features to time series
        ts_covF = TimeSeries.from_dataframe(df_feat, fill_missing_dates=True, freq=None)


        # create train and test split
        ts_train, ts_test = ts_P.split_after(SPLIT)
        covF_train, covF_test = ts_covF.split_after(SPLIT)

        scalerP = Scaler()
        scalerP.fit_transform(ts_train)
        ts_ttrain = scalerP.transform(ts_train)
        ts_ttest = scalerP.transform(ts_test)
        ts_t = scalerP.transform(ts_P)

        # make sure data are of type float
        ts_t = ts_t.astype(np.float32)
        ts_ttrain = ts_ttrain.astype(np.float32)
        ts_ttest = ts_ttest.astype(np.float32)

        # do the same for features
        scalerF = Scaler()
        scalerF.fit_transform(covF_train)
        covF_ttrain = scalerF.transform(covF_train)
        covF_ttest = scalerF.transform(covF_test)
        covF_t = scalerF.transform(ts_covF)

        # make sure data are of type float
        covF_ttrain = ts_ttrain.astype(np.float32)
        covF_ttest = ts_ttest.astype(np.float32)
        covF_t = covF_t.astype(np.float32)

        # add monthly indicators
        covT = datetime_attribute_timeseries(ts_P.time_index,
                                                attribute="month",
                                                one_hot=False)


        # train/test split
        covT_train, covT_test = covT.split_after(SPLIT)

        # rescale the covariates: fitting on the training set
        scalerT = Scaler()
        scalerT.fit(covT_train)
        covT_ttrain = scalerT.transform(covT_train)
        covT_ttest = scalerT.transform(covT_test)
        covT_t = scalerT.transform(covT)

        covT_t = covT_t.astype(np.float32)

        model = VARIMA(p=12)
        model.fit(ts_ttrain)

        #model.save('models/test model.pth.tar')

        ts_tpred_long = model.predict(   n=31,
                                    num_samples=N_SAMPLES,
                                    n_jobs=N_JOBS,
                                    verbose=True)
        # mark the test set for evaluation
        ts_tpred = ts_tpred_long[:len(ts_test)]

        # take the rest of the predictions and transform them back into a dataframe
        ts_tfut = ts_tpred_long[len(ts_test):]

        # remove the scaler transform
        ts_tfut = scalerP.inverse_transform(ts_tfut)

        # convert to dataframe
        pred_row = ts_tfut.quantile_df()
        pred_row = pred_row.iloc[:,0].apply(lambda x: float(x.values))
        pred_row.name = pred_row.name.replace('_0.5','')

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


        # helper function: get forecast values for selected quantile q and insert them in dataframe dfY
        def predQ(ts_t, q):
            ts_tq = ts_t.quantile_timeseries(q)
            ts_q = scalerP.inverse_transform(ts_tq)
            s = TimeSeries.pd_series(ts_q)
            header = "Q" + format(int(q * 100), "02d")
            dfY[header] = s
            if q == 0.5:
                ts_q50 = ts_q
                q50_RMSE = rmse(ts_q50, ts_test)
                q50_MAPE = mape(ts_q50, ts_test)
                print("RMSE:", f'{q50_RMSE:.2f}')
                print("MAPE:", f'{q50_MAPE:.2f}')
                return [q50_RMSE, q50_MAPE]

        # call helper function predQ, once for every quantile
        perf_scores = [predQ(ts_tpred, q) for q in QUANTILES]
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

        pd.DataFrame(result_log).T.to_csv('result_logs/looped transformer model results '+
                                          date_run+
                                          '.csv')

        pred_df.to_csv('output/predicted job posting shares '+
                                          date_run+
                                          '.csv')

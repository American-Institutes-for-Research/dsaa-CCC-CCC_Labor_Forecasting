# adapting methods from
# https://towardsdatascience.com/transformer-unleashed-deep-forecasting-of-multivariate-time-series-in-python-9ca729dac019
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from datetime import datetime

def run_transformer_test(EPOCHS=200, pca_components = 30,N_SAMPLES = 100,DIM_FF = 128,HEADS = 4
                    ,ENCODE = 4, DECODE = 4 , BATCH = 32 ):
    '''
    params:
       EPOCS - number of epocs the model trains on
       pca_components - number of PCA components created
       N_SAMPLES - number of times a prediction is sampled from a probabilistic model
       DIM_FF - dimensions of the feedforward network
        HEADS -  The number of heads in the multi-head attention mechanism
        ENCODE - encoder layers
        DECODE - decoder layers
        BATCH - batch size
    Function to test run transformer model with various parameters, and understand runtime
    Just runs on a single target
    '''


    FEAT = 32           # d_model = number of expected features in the inputs, up to 512

    ACTF = "relu"       # activation function, relu (default) or gelu
    SCHLEARN = None     # a PyTorch learning rate scheduler; None = constant rate
    LEARN = 1e-3        # learning rate
    VALWAIT = 1         # epochs to wait before evaluating the loss on the test/validation set
    DROPOUT = 0.1       # dropout rate

    RAND = 42           # random seed
    N_JOBS = 3          # parallel processors to use;  -1 = all processors

    # default quantiles for QuantileRegression
    QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

    SPLIT = 0.9         # train/test %

    FIGSIZE = (9, 6)

    start = datetime.now()
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

    # set a variable to target
    # TODO: set up as forloop for all targets
    t = targets[1]

    # figure out what features to use
    features = df.corr()[t]
    # filter to only those with at least a moderate correlation of .25
    features = features.loc[features.abs()> .25]
    features = features.drop(t).index

    # min max scale features
    df_feat = df[features]
    df_feat = MinMaxScaler().fit_transform(df_feat)

    # run PCA to reduce number of features
    pca = PCA(n_components=min(pca_components, len(features)))
    res_pca = pca.fit_transform(df_feat)

    # collect principal components in a dataframe
    df_pca = pd.DataFrame(res_pca)
    df_pca.index = df.index
    df_pca = df_pca.add_prefix("pca")
    df_pca[t] = df[t]

    # select pcas with correlation >.10
    selected_pca = df_pca.corr()[t].loc[df_pca.corr()[t].abs() > .1].drop(t).index

    #df.corr().to_csv('data/test corr.csv')

    # convert target to time series
    ts_P = TimeSeries.from_series(df[t], fill_missing_dates=True, freq=None)
    #ts_P = pd.Series([i[0] for i in ts_P.values()])
    # convert features to time series
    ts_covF = TimeSeries.from_dataframe(df_pca[selected_pca], fill_missing_dates=True, freq=None)


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

    model = TransformerModel(
     #                   input_chunk_length = INLEN,
     #                   output_chunk_length = N_FC,
                        input_chunk_length= 12,
                        output_chunk_length= len(ts_ttest),
                        batch_size = BATCH,
                        n_epochs = EPOCHS,
                        model_name = "Transformer_test_skill",
                        nr_epochs_val_period = VALWAIT,
                        d_model = FEAT,
                        nhead = HEADS,
                        num_encoder_layers = ENCODE,
                        num_decoder_layers = DECODE,
                        dim_feedforward = DIM_FF,
                        dropout = DROPOUT,
                        activation = ACTF,
                        random_state=RAND,
                        likelihood=QuantileRegression(quantiles=QUANTILES),
                        optimizer_kwargs={'lr': LEARN},
                        add_encoders={"cyclic": {"future": ["month"]}},
                        save_checkpoints=True,
                        force_reset=True
                        )
    model.fit(ts_ttrain,
                    past_covariates=covF_ttrain,
                    verbose=True)

    model.save('models/test model.pth.tar')

    ts_tpred = model.predict(   n=len(ts_ttest),
                                num_samples=N_SAMPLES,
                                n_jobs=N_JOBS,
                                verbose=True)

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

    # move Q50 column to the left of the Actual column
    col = dfY.pop("Q50")
    dfY.insert(1, col.name, col)
    dfY.iloc[np.r_[0:2, -2:0]]

    # log results
    result_log = pd.Series()

    result_log['timestamp'] = datetime.now().strftime('%H_%M_%d_%m_%Y')
    result_log['runtime'] = datetime.now() - start
    result_log['RMSE'] = perf_scores[0]
    result_log['MAPE'] = perf_scores[1]
    result_log['num_features_raw'] = df.shape[1] - 2
    result_log['pca_components'] = pca_components
    result_log['num_features_pca'] = len(selected_pca)
    result_log['EPOCHS'] = EPOCHS
    result_log['DIM_FF'] = DIM_FF
    result_log['N_SAMPLES'] = N_SAMPLES
    result_log['HEADS'] = HEADS
    result_log['ENCODE'] = ENCODE
    result_log['DECODE'] = DECODE
    result_log['BATCH'] = BATCH

    if os.path.exists('../result_logs/test transformer model params.csv'):
        pd.DataFrame(result_log).T.to_csv('result_logs/test transformer model params.csv', mode = 'a', header=False)
    else:
        pd.DataFrame(result_log).T.to_csv('result_logs/test transformer model params.csv')
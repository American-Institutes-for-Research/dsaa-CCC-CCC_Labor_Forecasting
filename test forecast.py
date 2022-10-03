# adapting methods from
# https://towardsdatascience.com/transformer-unleashed-deep-forecasting-of-multivariate-time-series-in-python-9ca729dac019

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

# set params
LOAD = False         # True = load previously saved model from disk?  False = (re)train the model
# SAVE = "\_TForm_model10e.pth.tar"   # file name to save the model under

EPOCHS = 200
INLEN = 32          # input size
FEAT = 32           # d_model = number of expected features in the inputs, up to 512
HEADS = 4           # default 8
ENCODE = 4          # encoder layers
DECODE = 4          # decoder layers
DIM_FF = 128        # dimensions of the feedforward network, default 2048
BATCH = 32          # batch size
ACTF = "relu"       # activation function, relu (default) or gelu
SCHLEARN = None     # a PyTorch learning rate scheduler; None = constant rate
LEARN = 1e-3        # learning rate
VALWAIT = 1         # epochs to wait before evaluating the loss on the test/validation set
DROPOUT = 0.1       # dropout rate
N_FC = 1            # output size

RAND = 42           # random seed
N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
N_JOBS = 3          # parallel processors to use;  -1 = all processors

# default quantiles for QuantileRegression
QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

SPLIT = 0.9         # train/test %

FIGSIZE = (9, 6)


df = pd.read_csv('data/test monthly counts 09302022.csv')
df = df.rename({'Unnamed: 0':'date'}, axis=1)
df['month']= df['date'].str[5:7].astype('int')
job_counts = df['Postings count']
# 5-50 filter is to remove months with 0 obs
# TODO: double check these rows are still right with new counts data set
df = df.iloc[5:50,:1000].reset_index(drop=True)

# create times series index
df['date'] = pd.to_datetime(df['date'])
df = df.set_index(pd.DatetimeIndex(df['date']))

# establish target columns as ones with an average obs count over 100
targets = df.mean().loc[df.mean()>100].index

# set a variable to target
# TODO: set up as forloop for all targets
t = targets[0]

# figure out what features to use
features = df.corr()[t]
# filter to only those with at least a moderate correlation of .25
features = features.loc[features.abs()> .25]
features = features.drop(t).index

# min max scale features
df_feat = df[features]
df_feat = MinMaxScaler().fit_transform(df_feat)

# run PCA to reduce number of features
pca = PCA(n_components=30)
res_pca = pca.fit_transform(df_feat)

# collect principal components in a dataframe
df_pca = pd.DataFrame(res_pca)
df_pca.index = df.index
df_pca = df_pca.add_prefix("pca")
df_pca[t] = df[t]

# select pcas with correlation >.10
selected_pca = df_pca.corr()[t].loc[df_pca.corr()[t].abs() > .1].drop(t).index

#df.corr().to_csv('data/test corr.csv')

# convert features to time series
ts_P = TimeSeries.from_series(df[t], fill_missing_dates=True, freq=None)
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

print("first and last row of scaled price time series:")
pd.options.display.float_format = '{:,.2f}'.format
ts_t.pd_dataframe().iloc[[0,-1]]


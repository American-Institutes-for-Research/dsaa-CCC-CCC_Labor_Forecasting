import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

df = pd.read_csv('data/test monthly counts.csv')
df = df.rename({'Unnamed: 0':'date'}, axis=1)
df['month']= df['date'].str[5:7].astype('int')
job_counts = df['Postings count']
# 5-50 filter is to remove months with 0 obs
df = df.iloc[5:50,:1000].reset_index(drop=True)

# establish target columns as ones with an average obs count over 100
targets = df.mean().loc[df.mean()>100].index

# set a variable to target
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
df_pca = df_pca.add_prefix("pca")
df_pca[t] = df[t]
# select pcas with correlation >.10
selected_pca = df_pca.corr()[t].loc[df_pca.corr()[t].abs() > .1].drop(t)

ts_P = TimeSeries.from_series(df[t])

df.corr().to_csv('data/test corr.csv')
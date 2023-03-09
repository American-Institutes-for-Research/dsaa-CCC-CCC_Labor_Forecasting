import pandas as pd
from darts.models.forecasting.linear_regression_model import LinearRegressionModel
from darts.models import TransformerModel
from darts import TimeSeries
from darts.utils.likelihood_models import QuantileRegression
from darts.models.forecasting.xgboost import XGBModel
import matplotlib.pyplot as plt
from darts.models import NBEATSModel

df = pd.read_excel('data/dummy data.xlsx')
#df = pd.read_excel('data/dummy data nonlinear.xlsx')
#df = pd.read_excel('data/dummy data nonlinear2.xlsx')
df = df.set_index(pd.to_datetime(df['dates']))
y = TimeSeries.from_series(df['y'])
train = y[:-36]
test = y[-36:]

#X = TimeSeries.from_dataframe(df[['x1','x2','x3','year','month']])
#X = TimeSeries.from_dataframe(df[['year','month']])

model = LinearRegressionModel(
    lags=24,
    #lags_past_covariates=6,
    output_chunk_length=36
)
model.fit(train)
preds = model.predict(n=36, series=test)

model2 = XGBModel(
    lags=24,
    output_chunk_length=36
)

model2.fit(train)
preds2 = model2.predict(n=36, series=test)
result_df = pd.concat([train.pd_series(),test.pd_series(), preds.pd_series(), preds2.pd_series()], axis=1)
result_df.columns = ['actual train','actual test','linear','XGB']
result_df.plot()
plt.show()
# EPOCHS=200
# N_SAMPLES = 100
# DIM_FF = 128
# HEADS = 4
# ENCODE = 4
# DECODE = 4
# BATCH = 32
# FEAT = 32  # d_model = number of expected features in the inputs, up to 512
#
# ACTF = "relu"  # activation function, relu (default) or gelu
# SCHLEARN = None  # a PyTorch learning rate scheduler; None = constant rate
# LEARN = 1e-3  # learning rate
# VALWAIT = 1  # epochs to wait before evaluating the loss on the test/validation set
# DROPOUT = 0.1  # dropout rate
#
# RAND = 42  # random seed
# N_JOBS = 3
# QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
#
# model2 = TransformerModel(
#                     input_chunk_length=12,  # originally 12
#                     output_chunk_length=12,
#                     batch_size=BATCH,
#                     n_epochs=EPOCHS,
#                     model_name="Transformer_test_skill",
#                     nr_epochs_val_period=VALWAIT,
#                     d_model=FEAT,
#                     nhead=HEADS,
#                     num_encoder_layers=ENCODE,
#                     num_decoder_layers=DECODE,
#                     dim_feedforward=DIM_FF,
#                     dropout=DROPOUT,
#                     activation=ACTF,
#                     random_state=RAND,
#                     optimizer_kwargs={'lr': LEARN},
#                     add_encoders={"cyclic": {"future": ["month"]}},
#                     save_checkpoints=True,
#                     force_reset=True
# )
#
# model2.fit(y)
# preds2 = model2.predict(n=30, series = y)

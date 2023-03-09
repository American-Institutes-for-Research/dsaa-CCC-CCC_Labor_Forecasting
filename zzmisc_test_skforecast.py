import pandas as pd
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.neural_network import MLPRegressor
from skforecast.model_selection import grid_search_forecaster

#df = pd.read_excel('data/dummy data.xlsx')
df = pd.read_excel('data/dummy data nonlinear.xlsx')
#df = pd.read_excel('data/dummy data nonlinear2.xlsx')
df = df.set_index(pd.to_datetime(df['dates']))
y = df['y']
train = y.iloc[:-36]
test = y.iloc[-36:]


forecaster = ForecasterAutoreg(
                regressor = MLPRegressor(),
                lags = 12
             )

# Lags used as predictors
lags_grid = [10, 20]

# Regressor's hyperparameters
param_grid = {'hidden_layer_sizes':[(100,),(1000,)],
              'activation':['identity', 'logistic', 'tanh', 'relu'],
              'solver':['lbfgs', 'sgd', 'adam'],
              'alpha':[0.0001,.001, .01, .00001,.00005, .005, .05],
              'learning_rate_init':[0.001,.0001, .0005,.005,.00001],
              'max_iter':[200,2000,20000]
}
results_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = train,
                        param_grid         = param_grid,
                        steps              = 36,
                        refit              = True,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(train)*0.5),
                        fixed_train_size   = False,
                        return_best        = True,
                        verbose            = False
               )

forecaster.fit(y=train)
preds = forecaster.predict(steps=36)
preds.index = test.index
result_df = pd.concat([train,test, preds], axis=1)
result_df.columns = ['actual train','actual test','MLP']
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

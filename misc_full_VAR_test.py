# scratch file to replicate errors obtained by running a VAR model with all 22k skills
import pandas as pd
from statsmodels.tsa.api import VAR
import numpy as np

df = pd.read_csv('data/test monthly counts season-adj.csv', index_col=0)
date_idx = pd.to_datetime(df.index)
df = df.set_index(pd.DatetimeIndex(date_idx))
model = VAR(df)
max_lags = 12

fit_results = pd.Series()
for i in range(1,max_lags+1):
    model_cand = model.fit(i)
    try:
        fit_results.loc[i] = model_cand.aic
    except np.linalg.LinAlgError:
        pass

if len(fit_results) > 0:
    p_order = fit_results.idxmax()
else:
    p_order = max_lags
model_fitted = model.fit(p_order)
model_fitted.summary()
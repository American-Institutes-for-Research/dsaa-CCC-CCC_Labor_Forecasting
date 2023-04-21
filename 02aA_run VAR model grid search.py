'''
Luke Patterson
02a_run VAR model.py

Purpose: run VAR forecasting model with a variety of parameters

'''
from model_VAR_forecast_loop import run_VAR_loop
import itertools
import pandas as pd
from utils import grid_search

grid_search(
    params_grid= {
        'trend': ['c', 'ct', 'ctt', 'n'],
        'max_lags': [1, 3, 6, 9, 12, 18],
        'cand_features_num': [3, 5, 10, 20, 30],
        'ic': ['aic', 'fpe', 'hqic', 'bic', None]
    },
    default_params= {
        'hierarchy_lvl':'subcategory',
        'ccc_taught_only': False,
        'min_tot_inc':0,
        'min_month_avg':0,
        'max_diffs':0,
        'test_tvalues':12
    },
    loop_func= run_VAR_loop,
    batch_name = 'VAR grid search 12mo test subcat'
)

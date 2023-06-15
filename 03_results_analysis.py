'''
Luke Patterson
03_results analysis.py

Purpose: Produce a results summary file for a particular model forecasting run
input:
    'result_logs/looped VAR model results '+ date_run+' '+run_name +'.csv' <- Log of parameters and performance metrics
    'output/predicted job posting shares '+date_run+' '+run_name+'.csv') <- Forecasted time series
output:
    'output/predicted changes '+date_run+' '+run_name+'.csv') <- results summary
'''
import pandas as pd
from utils import results_analysis
import os

# for file in os.listdir('output/batch_ARIMA top grid search runs'):
#     results_analysis(file.replace('.csv',''),
#                      fcast_folder='output/batch_ARIMA top grid search runs/')

# for file in os.listdir('output/batch_VAR top grid search runs cat_subcat'):
#     results_analysis(file.replace('.csv',''),
#                      fcast_folder='output/batch_VAR top grid search runs cat_subcat/')

for file in os.listdir('output/batch_VAR top grid search runs'):
    results_analysis(file.replace('.csv',''),
                     fcast_folder='output/batch_VAR top grid search runs/')


pass

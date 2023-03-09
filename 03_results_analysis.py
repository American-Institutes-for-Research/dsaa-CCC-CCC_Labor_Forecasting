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


results_analysis('predicted job posting shares 14_54_06_03_2023 DLPM 1 month input lvl subcategory', panel_data=True)
#results_analysis('predicted job posting shares 12_20_06_03_2023 Transformer 3.5 yr out lvl category')
pass

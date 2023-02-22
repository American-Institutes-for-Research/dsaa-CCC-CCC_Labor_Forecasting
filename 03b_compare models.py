'''
Luke Patterson
03b_compare models.py

Purpose: Produce comparisons across multiple models for the forecasted and actual values for skills
input:
    'result_logs/looped VAR model results '+ date_run+' '+run_name +'.csv' <- Log of parameters and performance metrics
    'output/predicted job posting shares '+date_run+' '+run_name+'.csv') <- Forecasted time series
output:
    'output/exhibits/'+date_run+' '+run_name/+'*.png') <- visualization graph files
'''

from utils import compare_results

compare_results(
    runnames=[
        'predicted job posting shares 15_29_15_02_2023 DLPM 6 month input lvl category',
        'predicted job posting shares 09_46_26_01_2023 VAR no differencing 6 month input lvl category',
        'predicted job posting shares 10_41_26_01_2023 Transformer no differencing 6 month input lvl category'
    ],
    labels=[
        'panel',
        'VAR',
        'transformer'
    ],
    panel_indicators=[
        True,
        False,
        False
    ],
    title = 'VAR_DLPM_ML comparison',
    hierarchy_lvl= 'category'
)

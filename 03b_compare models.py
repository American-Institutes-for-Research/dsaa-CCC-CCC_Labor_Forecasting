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
        'predicted job posting shares 14_54_06_03_2023 DLPM 1 month input lvl subcategory',
        'predicted job posting shares 14_33_06_03_2023 VAR no differencing 1 month input 3.5 year output lvl subcategory',
        'predicted job posting shares 14_50_06_03_2023 Transformer 3.5 yr out lvl subcategory'
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
    title = 'VAR_DLPM_ML comparison subcat',
    hierarchy_lvl= 'subcategory',
    sample = 20
)

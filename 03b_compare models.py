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
        # 'predicted job posting shares 14_54_06_03_2023 DLPM 1 month input lvl subcategory',
        'predicted job posting shares 14_33_06_03_2023 VAR no differencing 1 month input 3.5 year output lvl subcategory',
        'predicted job posting shares 16_06_15_03_2023 ARIMA test lvl subcategory',
        'predicted job posting shares 17_16_14_03_2023 Neuralprophet AR yhat1 test lvl subcategory'
    ],
    labels=[
        # 'panel',
        'VAR',
        'ARIMA',
        'ProphetAR'
    ],
    panel_indicators=[
        # True,
        False,
        False,
        False
    ],
    title = 'VAR_Prophet_ARIMA comparison subcat',
    hierarchy_lvl= 'subcategory',
    #sample = 20
)
#
# compare_results(
#     runnames=[
#         'predicted job posting shares 09_21_28_02_2023 DLPM 6 month input lvl category',
#         'predicted job posting shares 13_22_14_03_2023 VAR no differencing 6 month input corrected data lvl category',
#         'predicted job posting shares 09_41_09_03_2023 Transformer 6 month input 12 output chunk len lvl category',
#         'predicted job posting shares 17_02_14_03_2023 Neuralprophet AR yhat1 test lvl category'
#     ],
#     labels=[
#         'panel',
#         'VAR',
#         'transformer',
#         'ProphetAR'
#     ],
#     panel_indicators=[
#         True,
#         False,
#         False,
#         False
#     ],
#     title = 'VAR_DLPM_ML_ProphAR comparison cat',
#     hierarchy_lvl= 'category'
# )

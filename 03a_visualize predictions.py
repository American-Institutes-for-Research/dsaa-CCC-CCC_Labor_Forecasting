'''
Luke Patterson
03a_visualize predictions.py

Purpose: Produce visualizations of the forecasted and actual values for skills for a particular model run
input:
    'result_logs/looped VAR model results '+ date_run+' '+run_name +'.csv' <- Log of parameters and performance metrics
    'output/predicted job posting shares '+date_run+' '+run_name+'.csv') <- Forecasted time series
output:
    'output/exhibits/'+date_run+' '+run_name/+'*.png') <- visualization graph files
'''
import pandas as pd
from matplotlib import pyplot as plt

from utils import visualize_predictions

#visualize_predictions('predicted job posting shares 11_48_22_12_2022 no filtered skills VAR')
#visualize_predictions('predicted job posting shares 09_42_29_12_2022 1 month input length')
#visualize_predictions('predicted job posting shares 14_14_05_01_2023 ML differencing test')
# visualize_predictions('predicted job posting shares 14_54_10_01_2023 xgboost loop')
# visualize_predictions('predicted job posting shares 11_36_12_01_2023 no filtered skills VAR v2 - Copy')
# visualize_predictions('predicted job posting shares 13_03_12_01_2023 no filtered skills VAR v2 - Copy')
# visualize_predictions('predicted job posting shares 17_58_12_01_2023 filtered skills VAR no difference')
#visualize_predictions('predicted job posting shares 15_51_25_01_2023 VAR no differencing lvl category', sample=None)
visualize_predictions('predicted job posting shares 11_19_08_03_2023 DLPM 6 month input 12 output chunk len lvl category', sample=None, panel_data= True)
#visualize_predictions('predicted job posting shares 12_20_06_03_2023 Transformer 3.5 yr out lvl category', sample= None)

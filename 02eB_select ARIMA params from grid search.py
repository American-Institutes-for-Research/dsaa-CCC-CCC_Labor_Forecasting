'''
02aB_select ARIMA params from grid search.py
Luke Patterson
3/28/2023

Purposes: Go through ARIMA grid search results and select the model with the best RMSE
'''

import pandas as pd
import os

logfiles = os.listdir('result_logs/batch_ARIMA grid search 12 month test')
logfiles = [i for i in logfiles if 'looped ARIMA model ' in i]

result_df = pd.DataFrame()
for f in logfiles:
    df = pd.read_csv('result_logs/batch_ARIMA grid search 12 month test/'+f, index_col=0)
    # not all files got to all series, so we'll just
    row = pd.Series()
    if 'Normalized RMSE' in df.columns and df.shape[0]>1:
        row['filename'] = f
        row['RMSE'] = df['Normalized RMSE'].mean()
        for col in ['input_len_used', 'differences_made',
           'cand_features_num', 'auto_reg', 'moving_avg', 'trend', 'use_exog']:
            row[col] = df[col].iloc[0]
        row['cand_features_num'] = df['cand_features_num'].iloc[0]
        result_df = result_df.append(row, ignore_index=True)

result_df = result_df.sort_values('RMSE')
result_df.to_csv('result_logs/batch_ARIMA grid search 12 month test/RMSE summary.csv')

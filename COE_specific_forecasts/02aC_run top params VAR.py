'''
02aC_ run top params VAR.py
run top 10 performing grid search params at each hierarchy level
'''
import os
basepath = 'C:/AnacondaProjects/CCC forecasting'
os.chdir(basepath)

import pandas as pd
from model_VAR_forecast_loop import run_VAR_loop

param_df = pd.read_csv("result_logs/batch_VAR grid search/RMSE summary.csv")
param_cols = ['trend','max_lags','cand_features_num']

param_dicts = []
for i, row in param_df.head(20).iterrows():
    param_dict = {}
    for p in param_cols:
        param_dict[p] = row[p]

    param_dicts.append(param_dict)
param_dicts = param_dicts[14:]

coe_names = ['Business','Construction','Culinary & Hospitality','Education & Child Development','Engineering & Computer Science',
             'Health Science','Information Technology','Manufacturing','Transportation, Distribution, & Logistics']
for coe in coe_names:
    print(coe)
    for lvl in ['category','subcategory', 'skill']:
        print(lvl)
        for n, params in enumerate(param_dicts):
            try:
                print('param set', n)
                print(params)
                run_VAR_loop(hierarchy_lvl=lvl, ccc_taught_only=False, max_diffs=0,
                             run_name='COE '+coe+' VAR run #'+str(n), batch_name='COE VAR runs',
                             analyze_results=False, test_tvalues=5,
                             custom_input_data_path= 'data/test monthly counts season-adj ' + coe + ' ' + lvl +'.csv',
                             **params)
            except Exception as e:
                print('error with params:', params)
                print('error message:', e)
                continue
pass

'''
02aC_ run top params VAR.py
run top 10 performing grid search params at each hierarchy level
'''

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
#for lvl in ['category','subcategory', 'skill']:
for lvl in ['skill']:
    print(lvl)
    for n1, params in enumerate(param_dicts):
        n= n1 +14
        try:
            print('param set', n)
            print(params)
            run_VAR_loop(hierarchy_lvl=lvl, ccc_taught_only=False, max_diffs=0,
                           run_name='VAR top grid search runs #'+str(n), batch_name='VAR top grid search runs',
                           analyze_results=False, test_tvalues=5, **params)
        except Exception as e:
            print('error with params:', params)
            print('error message:', e)
            #err_row = pd.Series(p)
            #err_row['error'] = e
            #error_tracker = error_tracker.append(err_row, ignore_index=True)
            continue
pass

pass

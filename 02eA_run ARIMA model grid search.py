import pandas as pd

from model_statsmodels_arima_forecast_loop import run_ARIMA_loop
import itertools

params_grid = {
    'cand_features_num': [3,5,10,20,30],
    'auto_reg': [1,3,5,7,9,12,18],
    'moving_avg': [1,3,6,9,12,18],
    'trend':[None,'n','c','t','ct',(1,1,1),(1,1,1,1)],
    'use_exog':[False,True]
}
keys, values = zip(*params_grid.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

error_tracker = pd.DataFrame()

for n, p in enumerate(permutations_dicts):
    print('iteration', n, 'of',len(permutations_dicts))
    try:
        run_ARIMA_loop(hierarchy_lvl='category', ccc_taught_only=False, max_diffs=0, run_name='ARIMA grid search 12 month test', batch_name='ARIMA grid search 12 month test',
                       analyze_results=False, test_tvalues = 12, **p)
    except Exception as e:
        print('error with params:',p)
        print('error message:',e)
        err_row = pd.Series(p)
        err_row['error'] = e
        error_tracker = error_tracker.append(err_row, ignore_index=True)
        continue

error_tracker.to_csv('result_logs/batch_ARIMA grid test/error_tracker.csv')

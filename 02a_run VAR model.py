'''
Luke Patterson
02a_run VAR model.py

Purpose: run VAR forecasting model with a variety of parameters

'''
from model_VAR_forecast_loop import run_VAR_loop
import pandas as pd

# run_VAR_loop(min_tot_inc=0, min_month_avg=0, ccc_taught_only=False, run_name='no filtered skills VAR v2')
# run_VAR_loop(min_tot_inc=0, min_month_avg=0, max_diffs=0, ccc_taught_only=False, hierarchy_lvl='category', run_name='VAR no differencing')
# run_VAR_loop(min_tot_inc=0, min_month_avg=0, max_diffs=0,  ccc_taught_only=False, hierarchy_lvl='subcategory', run_name='VAR no differencing')
# run_VAR_loop(max_lags=6, input_len_used=6,min_tot_inc=0, min_month_avg=0, max_diffs=0, ccc_taught_only=False,
#              hierarchy_lvl='category', run_name='VAR no differencing 6 month input')
# run_VAR_loop(max_lags=6, input_len_used=6,min_tot_inc=0, min_month_avg=0, max_diffs=0, ccc_taught_only=False,
#              hierarchy_lvl='category', run_name='VAR no differencing 6 month input no season adj', season_adj=False)
# run_VAR_loop(max_lags=6, input_len_used=6, min_tot_inc=0, min_month_avg=0, max_diffs=0,  ccc_taught_only=False,
#              hierarchy_lvl='subcategory', run_name='VAR no differencing 6 month input')

# run_VAR_loop(max_lags=6, input_len_used=6, min_tot_inc=0, min_month_avg=0, max_diffs=0,  ccc_taught_only=False, test_tvalues=12,
#              hierarchy_lvl='subcategory', run_name='VAR 12 month test')
# run_VAR_loop(max_lags=1, input_len_used=1, min_tot_inc=0, min_month_avg=0, max_diffs=0,  ccc_taught_only=False,
#              hierarchy_lvl='subcategory', run_name='VAR no differencing 1 month input 3.5 year output')
# run_VAR_loop(max_lags=9, min_tot_inc=0, min_month_avg=0, max_diffs=0, cand_features_num=20,trend='n', test_tvalues=12,
#              ccc_taught_only=False, hierarchy_lvl='category', run_name='VAR grid search optimal 12mo')
# run_VAR_loop(max_lags=9, min_tot_inc=0, min_month_avg=0, max_diffs=0, cand_features_num=20,trend='n', test_tvalues=12,
#              ccc_taught_only=False, hierarchy_lvl='subcategory', run_name='VAR grid search optimal 12mo')

# run_VAR_loop(max_lags=6, input_len_used=6,min_tot_inc=0, min_month_avg=0, max_diffs=0, ccc_taught_only=False,
#              hierarchy_lvl='category', run_name='VAR for presentation', season_adj=False)
# run_VAR_loop(max_lags=6, input_len_used=6, min_tot_inc=0, min_month_avg=0, max_diffs=0,  ccc_taught_only=False,
#              hierarchy_lvl='subcategory', run_name='VAR for presentation')
# run_VAR_loop(max_lags=6, input_len_used=6, min_tot_inc=50, min_month_avg=50, max_diffs=0,  ccc_taught_only=False,
#              hierarchy_lvl='skill', run_name='VAR for presentation')


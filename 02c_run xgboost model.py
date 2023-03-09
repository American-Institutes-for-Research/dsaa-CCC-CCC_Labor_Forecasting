'''
Luke Patterson
02c_run xgboost model.py

Purpose: run xgboost forecasting model with a variety of parameters

'''
from xgboost_forecast_loop import run_xgboost_loop

run_xgboost_loop(run_name='xgboost loop', hierarchy_lvl='category', ccc_taught_only='False', visualize_results=True)

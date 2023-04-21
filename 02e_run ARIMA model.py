from model_statsmodels_arima_forecast_loop import run_ARIMA_loop

# run_ARIMA_loop(hierarchy_lvl='category', ccc_taught_only=False, max_diffs=0, run_name='ARIMA for presentation')
# run_ARIMA_loop(hierarchy_lvl='subcategory', ccc_taught_only=False, max_diffs=0, run_name='ARIMA for presentation')
run_ARIMA_loop(hierarchy_lvl='skill', ccc_taught_only=False, max_diffs=0, run_name='ARIMA for presentation')
# run_ARIMA_loop(hierarchy_lvl='category', ccc_taught_only=False, max_diffs=0,
#                cand_features_num= 5,
#                input_len_used=12, auto_reg=9, moving_avg=6, trend='c',test_tvalues=12, use_exog=False,
#                run_name='ARIMA grid search optimal')
# run_ARIMA_loop(hierarchy_lvl='subcategory', ccc_taught_only=False, max_diffs=0,
#                cand_features_num= 5,
#                input_len_used=12, auto_reg=9, moving_avg=6, trend='c', test_tvalues=12, use_exog=False,
#                run_name='ARIMA grid search optimal v2')


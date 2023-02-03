# run VAR forecast with no filtered skills
from VAR_forecast_loop import run_VAR_loop

# run_VAR_loop(min_tot_inc=0, min_month_avg=0, ccc_taught_only=False, run_name='no filtered skills VAR v2')
# run_VAR_loop(min_tot_inc=0, min_month_avg=0, max_diffs=0, ccc_taught_only=False, hierarchy_lvl='category', run_name='VAR no differencing')
# run_VAR_loop(min_tot_inc=0, min_month_avg=0, max_diffs=0,  ccc_taught_only=False, hierarchy_lvl='subcategory', run_name='VAR no differencing')
run_VAR_loop(max_lags=6, input_len_used=6,min_tot_inc=0, min_month_avg=0, max_diffs=0, ccc_taught_only=False,
             hierarchy_lvl='category', run_name='VAR no differencing 6 month input')
run_VAR_loop(max_lags=6, input_len_used=6, min_tot_inc=0, min_month_avg=0, max_diffs=0,  ccc_taught_only=False,
             hierarchy_lvl='subcategory', run_name='VAR no differencing 6 month input')

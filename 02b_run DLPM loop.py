'''
Luke Patterson
02b_run DLPM model.py

Purpose: run dynamic linear panel forecasting model with a variety of parameters

'''
from model_dlpm_forecast_loop import run_DLPM_loop

# run_DLPM_loop(run_name='DLPM initial run')
# run_DLPM_loop(run_name='DLPM 3 month input',input_len_used=3, hierarchy_lvl='category')
# run_DLPM_loop(run_name='DLPM 3 month input',input_len_used=3, hierarchy_lvl='subcategory')

# run_DLPM_loop(run_name='DLPM 6 month input',input_len_used=6, hierarchy_lvl='subcategory')

# run_DLPM_loop(run_name='DLPM 6 month input',input_len_used=6, hierarchy_lvl='skill', ccc_taught_only=False
#               ,min_month_avg = 0, min_tot_inc = 0)

run_DLPM_loop(run_name='DLPM 6 month input 12 output chunk len',input_len_used=6,
    output_chunk_length= 3,pred_length= 12, hierarchy_lvl='category')

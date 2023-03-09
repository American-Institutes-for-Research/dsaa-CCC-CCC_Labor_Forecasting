'''
Luke Patterson
02a_run transformer model.py

Purpose: run ML transformer forecasting model with a variety of parameters

'''
from transformer_forecast_loop import run_transformer_loop

#run_transformer_loop(min_tot_inc=0, min_month_avg=0,run_name='no filtered skills')
# run_transformer_loop(min_tot_inc=0, min_month_avg=0,run_name='no filtered skills pt 2', start_val= 1173)
#run_transformer_loop(ccc_taught_only= False, min_tot_inc=0, min_month_avg=0,run_name='no filtered skills pt 3', start_val= 9091)
run_transformer_loop(ccc_taught_only= False, min_tot_inc=0, min_month_avg=0,run_name='no filtered skills test', start_val= 9091 + 8325)

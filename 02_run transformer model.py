'''
Luke Patterson
02_run transformer model.py

Purpose: run ML transformer forecasting model with a variety of parameters

'''

from transformer_forecast_loop import run_transformer_loop

# run_transformer_loop(hierarchy_lvl='category', ccc_taught_only=False)
# run_transformer_loop(hierarchy_lvl='subcategory', ccc_taught_only=False)
# run_transformer_loop(input_len_used=1, min_tot_inc=0, min_month_avg=0,  ccc_taught_only=False,
#              hierarchy_lvl='category', run_name='Transformer no differencing 1 month input')
run_transformer_loop(input_len_used=1, min_tot_inc=0, min_month_avg=0, ccc_taught_only=False,
             hierarchy_lvl='subcategory',pred_length=42, run_name='Transformer 3.5 yr out')

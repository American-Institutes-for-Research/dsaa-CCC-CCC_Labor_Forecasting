'''
Luke Patterson
02_run transformer model_MM.py

Purpose: run ML transformer forecasting model with a variety of parameters

'''

from transformer_forecast_loop_MM_old import run_transformer_loop_MM
print('function imported')

run_transformer_loop_MM(EPOCHS=500, DIM_FF=2048, ENCODE=32, DECODE=32, SPLIT=.75, start_val=19, end_val=20,
                        ccc_taught_only=False, hierarchy_lvl='skill', run_name='MM', exit_loop=False, #min_tot_inc=0, min_month_avg=0,
                        features_print=True, features_export=False)
# run_transformer_loop(hierarchy_lvl='subcategory', ccc_taught_only=False)
# run_transformer_loop(input_len_used=1, min_tot_inc=0, min_month_avg=0,  ccc_taught_only=False,
#              hierarchy_lvl='category', run_name='Transformer no differencing 1 month input')
# run_transformer_loop(input_len_used=1, min_tot_inc=0, min_month_avg=0, ccc_taught_only=False,
#              hierarchy_lvl='subcategory',pred_length=42, run_name='Transformer 3.5 yr out')
# run_transformer_loop(input_len_used=1, min_tot_inc=0, min_month_avg=0,  ccc_taught_only=False,
#              hierarchy_lvl='category', run_name='Transformer no differencing 1 month input')

# run_transformer_loop(run_name='Transformer 6 month input 12 output chunk len',input_len_used=6,
#     output_chunk_len= 12, hierarchy_lvl='category')

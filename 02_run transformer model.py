from multivar_forecast_loop import run_transformer_loop

# run_transformer_loop(hierarchy_lvl='category', ccc_taught_only=False)
# run_transformer_loop(hierarchy_lvl='subcategory', ccc_taught_only=False)
run_transformer_loop(input_len_used=6, min_tot_inc=0, min_month_avg=0,  ccc_taught_only=False,
             hierarchy_lvl='category', run_name='Transformer no differencing 6 month input')
# run_transformer_loop(input_len_used=6, min_tot_inc=0, min_month_avg=0, ccc_taught_only=False,
#              hierarchy_lvl='subcategory', run_name='Transformer no differencing 6 month input')
# run_transformer_loop()

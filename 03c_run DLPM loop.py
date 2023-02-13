from dlpm_forecast_loop import run_DLPM_loop

# run_DLPM_loop(run_name='DLPM initial run')
run_DLPM_loop(run_name='DLPM 3 month input',input_len_used=3, hierarchy_lvl='category')
run_DLPM_loop(run_name='DLPM 3 month input',input_len_used=3, hierarchy_lvl='subcategory')
run_DLPM_loop(run_name='DLPM 6 month input',input_len_used=6, hierarchy_lvl='category')
run_DLPM_loop(run_name='DLPM 6 month input',input_len_used=6, hierarchy_lvl='subcategory')


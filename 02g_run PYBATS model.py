from model_pybats_forecast_loop import run_pybats_loop

run_pybats_loop(hierarchy_lvl='category', ccc_taught_only=False, test_split = .25, run_name='pybats prior model test',
                k=1, forecast_steps= 34, cand_features_num= 2)
# run_pybats_loop(hierarchy_lvl='subcategory', ccc_taught_only=False, test_split = .25, run_name='pybats grid search optimal',
#                 k=3,rho=.1, deltrend=.1, delregn=.5)

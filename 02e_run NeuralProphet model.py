from model_neuralprophet_forecast_loop import run_neuralprophet_loop

run_neuralprophet_loop(hierarchy_lvl='subcategory', ccc_taught_only=False, run_name='Neuralprophet AR yhat1 12 month test',past_months_data=12,
                       test_split=.25,)

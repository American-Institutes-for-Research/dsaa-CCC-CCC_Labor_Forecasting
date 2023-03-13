# run transformer with no filtered skills
from model_transformer_forecast_loop import run_transformer_loop

#run_transformer_loop(min_tot_inc=0, min_month_avg=0,run_name='no filtered skills')
run_transformer_loop(run_name='ML differencing test', differenced =True)

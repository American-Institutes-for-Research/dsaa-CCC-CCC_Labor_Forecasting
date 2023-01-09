# run VAR forecast with no filtered skills
from VAR_forecast_loop import run_VAR_loop

run_VAR_loop(min_tot_inc=0, min_month_avg=0,run_name='no filtered skills VAR')
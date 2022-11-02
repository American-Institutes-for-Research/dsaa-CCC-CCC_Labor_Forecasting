from multivar_forecast_loop_Kalman import run_kalman_loop, prepare_data
import pandas as pd
df, targets = prepare_data()

# pred_df = pd.read_csv('output/predicted job posting shares 16_28_13_10_2022.csv',index_col=0)
# result_log = pd.read_csv('result_logs/looped transformer model results 16_28_13_10_2022.csv', index_col=0).T

run_kalman_loop(df,targets)
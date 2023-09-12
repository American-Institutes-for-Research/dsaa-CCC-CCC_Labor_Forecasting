import pandas as pd
import os
pd.options.display.float_format = '{:.4f}'.format

old_path = 'C:/AnacondaProjects/CCC forecasting/result_logs/batch_VAR top grid search runs'
new_path = 'C:/AnacondaProjects/CCC forecasting/result_logs/batch_VAR top grid search runs 2023'
old_files = [i for i in os.listdir(old_path)
                if 'model results' in i and '.csv' in i]
new_files = [i for i in os.listdir(new_path)
                if 'model results' in i and '.csv' in i]


old_results = pd.Series()
for f in old_files:
    df = pd.read_csv(old_path + '/' + f)
    old_results = old_results.append(df['Normalized RMSE'])

new_results = pd.Series()
for f in new_files:
    df = pd.read_csv(new_path + '/' + f)
    new_results = new_results.append(df['Normalized RMSE'])

print(old_results.describe())
print(new_results.describe())

pass

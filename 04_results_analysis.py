import pandas as pd
from multivar_forecast_loop import prepare_data
df = pd.read_csv('output/predicted job posting shares 14_31_14_10_2022.csv', index_col = 0)

countdf, targets = prepare_data()

targets = list(targets)

# clean up some duplicated columns in the forecast
df = df.drop([i for i in df.columns if '1' in i], axis=1)
targets = [i for i in targets if i in df.columns]

countdf = countdf[targets]

lactual = countdf.iloc[-1,:]
pred_values = df.iloc[-3,:]


change = ((pred_values - lactual) / lactual) * 100

result_df = pd.concat([lactual,pred_values, change], axis=1)

result_df.columns = ['July 2022 actual', 'July 2024 predicted', 'Percent change']
result_df.index = [i.replace('Skill: ','') for i in result_df.index]

result_df = result_df.sort_values('Percent change', ascending=False)
result_df.to_csv('output/predicted changes.csv')


pass
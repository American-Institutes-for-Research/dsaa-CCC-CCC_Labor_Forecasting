import pandas as pd

files = {
    'Information Technology':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Information Technology formatted results 09212023.xlsx",
    'Health Science':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Health Science formatted results 09212023.xlsx",
    'Engineering & Computer Science':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Engineering & Computer Science formatted results 09212023.xlsx",
    'Education & Child Development':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Education & Child Development formatted results 09212023.xlsx",
    'Culinary & Hospitality':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Culinary & Hospitality formatted results 09212023.xlsx",
    'Construction':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Construction formatted results 09212023.xlsx",
    'Business':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Business formatted results 09212023.xlsx",
    'overall':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble multiple models formatted results 09212023.xlsx",
    'Transportation, Distribution, & Logistics':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Transportation, Distribution, & Logistics formatted results 09212023.xlsx",
    'Manufacturing':"C:\AnacondaProjects\CCC forecasting\output\exhibits\VAR_ARIMA ensemble Manufacturing formatted results 09212023.xlsx",
}
result_df = pd.DataFrame()
result_df2 = pd.DataFrame()
result_df3 = pd.DataFrame()
for key in files.keys():
    for cat in ['Category', 'Subcategory','Skill']:
        df = pd.read_excel(files[key], sheet_name= cat)
        if key == 'overall':
            df['std_est_ratio'] = df['Prediction std dev'] / df['July 2024 weighted predicted']
        else:
            df['std_est_ratio'] = df['Prediction std dev'] / df['July 2025 weighted predicted']
        result_df.loc[cat, key] = df['std_est_ratio'].mean()

result_df.to_excel('result_logs/std est ratio.xlsx')

for key in files.keys():
    for cat in ['Category', 'Subcategory','Skill']:
        df = pd.read_excel(files[key], sheet_name= cat)
        result_df2.loc[cat, key] = df.shape[0]
result_df2.to_excel('result_logs/num obs.xlsx')

for key in files.keys():
    for cat in ['Category', 'Subcategory','Skill']:
        df = pd.read_excel(files[key], sheet_name= cat)
        result_df3.loc[cat, key] = df['Average RMSE'].mean()
result_df3.to_excel('result_logs/mean error.xlsx')
pass

'''
Luke Patterson
01d_remove_seasonality.py

Purpose: For each of the hierarchy levels and county/MSA levels, read in the respective counts data set and remove
    seasonal variation from the time series. Seasonal variation is removed by decomposing each data series into
    trend, noise, and seasonal components using the seasonal_decompose function from the stats models package.
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.html

    Only the trend component is saved for each skill in the season-adj output files

input:
    data/test monthly counts [COE name] category.csv
    data/test monthly counts [COE name] subcategory.csv
    data/test monthly counts [COE name] skill.csv

output:
    data/test monthly counts season-adj county category.csv
    data/test monthly counts season-adj county subcategory.csv
    data/test monthly counts season-adj county skill.csv

'''
import os
basepath = 'C:/AnacondaProjects/CCC forecasting'
os.chdir(basepath)
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from dateutil.relativedelta import relativedelta

# check for seasonality.
def prepare_data(df):
    df = df.rename({'Unnamed: 0': 'date'}, axis=1)
    df['month'] = df['date'].str[5:7].astype('int')
    df = df.fillna(method='ffill')
    # 7-55 filter is to remove months with 0 obs
    df = df.iloc[7:55, :].reset_index(drop=True)

    # create times series index
    date_idx = pd.to_datetime(df['date'])

    # normalize all columns based on job postings counts
    df = df.drop('date', axis=1)

    job_counts = df['Postings count'].copy()
    raw_df = df.copy()
    df = df.divide(job_counts, axis=0)
    df['Postings count'] = job_counts
    df = df.set_index(pd.DatetimeIndex(date_idx))
    return df

def seasonality_loop(df, name, coe = None):


    df = prepare_data(df)
    targets = df.columns

    # todo: check sensitivity for not doing this expand this series
    # expand series out 6 months in either direction with same value so seasonality can be removed
    # for all values in the original series
    new_index = df.index.copy()
    for _ in range(6):
        new_index = new_index.union([new_index[-1]+ relativedelta(months=1)])
    new_df = pd.DataFrame(index=new_index)
    df = pd.concat([new_df, df], axis=1).ffill()
    for _ in range(6):
        new_index = new_index.union([new_index[0] - relativedelta(months=1)])
    new_df = pd.DataFrame(index=new_index)
    df = pd.concat([new_df, df], axis=1).bfill()

    clean_df = df.copy()

    # todo: sensitivity checks for generating this series including noise, and with different moving average periods
    for t in targets:
        print(t)
        if t != 'Postings count' and 't' != 'month':
            result = seasonal_decompose(df[t])
            clean_df[t] = result.trend

    clean_df = clean_df[targets]

    clean_df = clean_df.dropna()
    if coe is None:
        clean_df.to_csv('data/test monthly counts season-adj '+name+'.csv')
    else:
        clean_df.to_csv('data/test monthly counts season-adj ' + coe + ' ' + name + '.csv')

# remove seasonality for each of the three hierarchical levels
# df = pd.read_csv('data/test monthly counts.csv')
# seasonality_loop(df, 'skill')
# df2 = pd.read_csv('data/test monthly counts categories.csv')
# cat_df = df2[[i for i in df2.columns if 'Skill cat:' in i] + ['Unnamed: 0', 'Postings count']]
# seasonality_loop(cat_df, 'category')
# scat_df = df2[[i for i in df2.columns if 'Skill subcat:' in i] + ['Unnamed: 0', 'Postings count']]
# seasonality_loop(scat_df, 'subcategory')


# repeat all of
coe_names = ['Business','Construction','Culinary & Hospitality','Education & Child Development','Engineering & Computer Science',
             'Health Science','Information Technology','Manufacturing','Transportation, Distribution, & Logistics']

for c in coe_names:
    cdf = pd.read_csv('data/COE/test monthly counts '+c+' categories.csv')
    cdf2 = cdf[[i for i in cdf.columns if 'Skill:' in i] + ['Unnamed: 0', 'Postings count']]
    seasonality_loop(cdf2, 'skill', c)
    cat_df = cdf[[i for i in cdf.columns if 'Skill cat:' in i] + ['Unnamed: 0', 'Postings count']]
    seasonality_loop(cat_df, 'category', c)
    scat_df = cdf[[i for i in cdf.columns if 'Skill subcat:' in i] + ['Unnamed: 0', 'Postings count']]
    seasonality_loop(scat_df, 'subcategory', c)


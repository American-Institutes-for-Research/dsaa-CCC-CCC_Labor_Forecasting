import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from dateutil.relativedelta import relativedelta

# check for seasonality.
def seasonality_loop(df, name, county = None):
    df = df.rename({'Unnamed: 0':'date'}, axis=1)
    df['month']= df['date'].str[5:7].astype('int')
    df = df.fillna(method='ffill')
    # 7-55 filter is to remove months with 0 obs
    df = df.iloc[7:55,:].reset_index(drop=True)

    # create times series index
    date_idx = pd.to_datetime(df['date'])

    # normalize all columns based on job postings counts
    df = df.drop('date', axis=1)

    # export monthly average sample size
    if county is None:
        df.mean().to_csv('working/average monthly observations by counts '+name+'.csv')


    job_counts = df['Postings count'].copy()
    raw_df = df.copy()
    df = df.divide(job_counts, axis=0)
    df['Postings count'] = job_counts
    df = df.set_index(pd.DatetimeIndex(date_idx))

    targets = raw_df.columns

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

    for t in targets:
        print(t)
        if t != 'Postings count' and 't' != 'month':
            result = seasonal_decompose(df[t])
            clean_df[t] = result.trend

    clean_df = clean_df[targets]

    clean_df = clean_df.dropna()
    if county is None:
        clean_df.to_csv('data/test monthly counts season-adj '+name+'.csv')
    else:
        clean_df.to_csv('data/test monthly counts season-adj ' + county + ' ' + name + '.csv')

# remove seasonality for each of the three hierarchical levels
df = pd.read_csv('data/test monthly counts.csv')
seasonality_loop(df, 'skill')
df2 = pd.read_csv('data/test monthly counts categories.csv')
cat_df = df2[[i for i in df2.columns if 'Skill cat:' in i] + ['Unnamed: 0', 'Postings count']]
seasonality_loop(cat_df, 'category')
scat_df = df2[[i for i in df2.columns if 'Skill subcat:' in i] + ['Unnamed: 0', 'Postings count']]
seasonality_loop(scat_df, 'subcategory')


# repeat all of
county_names = ['Cook, IL', 'DuPage, IL', 'Lake, IL', 'Will, IL', 'Kane, IL',
       'Lake, IN', 'McHenry, IL', 'Kenosha, WI', 'Porter, IN', 'DeKalb, IL',
       'Kendall, IL', 'Grundy, IL', 'Jasper, IN', 'Newton, IN']

for c in county_names:
    cdf = pd.read_csv('data/test monthly counts ' +c+'.csv')
    seasonality_loop(cdf, 'skill', c)
    cdf2 = pd.read_csv('data/test monthly counts '+c+' categories.csv')
    cat_df = df2[[i for i in df2.columns if 'Skill cat:' in i] + ['Unnamed: 0', 'Postings count']]
    seasonality_loop(cat_df, 'category', c)
    scat_df = df2[[i for i in df2.columns if 'Skill subcat:' in i] + ['Unnamed: 0', 'Postings count']]
    seasonality_loop(scat_df, 'subcategory', c)


master_df = pd.DataFrame()
master_df2 = pd.DataFrame()
master_df3 = pd.DataFrame()
for c in county_names:
    df = pd.read_csv('data/test monthly counts season-adj ' + c + ' category.csv')
    df2 = pd.read_csv('data/test monthly counts season-adj ' + c + ' subcategory.csv')
    df3 = pd.read_csv('data/test monthly counts season-adj ' + c + ' skill.csv')
    df['county'] = c
    df2['county'] = c
    df3['county'] = c
    df = df.set_index(['county','date'])
    df2 = df2.set_index(['county','date'])
    df3 = df3.set_index(['county', 'date'])
    master_df = pd.concat([master_df, df])
    master_df2 = pd.concat([master_df2, df2])
    master_df3 = pd.concat([master_df3, df3])
master_df.to_csv('data/test monthly counts season-adj county category.csv')
master_df2.to_csv('data/test monthly counts season-adj county subcategory.csv')
master_df2.to_csv('data/test monthly counts season-adj county skill.csv')

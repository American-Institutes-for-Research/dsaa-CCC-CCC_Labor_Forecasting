import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from dateutil.relativedelta import relativedelta

# check for seasonality.
def seasonality_loop(df, name):
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
    clean_df.to_csv('data/test monthly counts season-adj '+name+'.csv')

df = pd.read_csv('data/test monthly counts.csv')
seasonality_loop(df, 'skill')
df2 = pd.read_csv('data/test monthly counts categories.csv')
cat_df = df2[[i for i in df2.columns if 'Skill cat:' in i] + ['Unnamed: 0', 'Postings count']]
seasonality_loop(cat_df, 'category')
scat_df = df2[[i for i in df2.columns if 'Skill subcat:' in i] + ['Unnamed: 0', 'Postings count']]
seasonality_loop(scat_df, 'subcategory')

import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.relativedelta import relativedelta

# load full data set to get columns of the panel data set
df = pd.read_csv('data/test monthly counts.csv', nrows = 1)
columns = [i for i in df.columns if 'Skill:' in i]

df = pd.DataFrame(columns=columns)

# load each county data and append to panel data
county_data = os.listdir('data/')
county_data = [i for i in county_data if ', I' in i]

for i in county_data:
    print(i)
    county_df = pd.read_csv('data/'+i)

    county_df = county_df.fillna(method='ffill')
    # 7-55 filter is to remove months with 0 obs
    county_df = county_df.iloc[7:55, :]

    # unnamed: 0 is date
    date_col = county_df['Unnamed: 0'].copy()
    date_idx = pd.DatetimeIndex(county_df['Unnamed: 0'])
    county_df = county_df.set_index(date_idx)

    county_df = county_df.drop('Unnamed: 0', axis=1)

    job_counts = county_df['Postings count'].copy()
    raw_df = county_df.copy()
    county_df = county_df.divide(job_counts, axis=0)
    county_df['Postings count'] = job_counts

    targets = raw_df.columns

    # expand series out 6 months in either direction with same value so seasonality can be removed
    # for all values in the original series
    new_index = county_df.index.copy()
    for _ in range(6):
        new_index = new_index.union([new_index[-1] + relativedelta(months=1)])
    new_df = pd.DataFrame(index=new_index)
    county_df = pd.concat([new_df, county_df], axis=1).ffill()
    for _ in range(6):
        new_index = new_index.union([new_index[0] - relativedelta(months=1)])
    new_df = pd.DataFrame(index=new_index)
    county_df = pd.concat([new_df, county_df], axis=1).bfill()

    clean_df = county_df.copy()

    for t in targets:
        if t != 'Postings count' and 't' != 'month':
            result = seasonal_decompose(county_df[t])
            clean_df[t] = result.trend

    clean_df = clean_df[targets]

    clean_df = clean_df.dropna()

    date_col.index = clean_df.index
    clean_df['date'] = date_col
    county_name = i.split()[3].replace(',', '')
    clean_df['county'] = county_name
    clean_df = clean_df.set_index(['county', 'date'], drop=True)

    df = df.append(clean_df)

df.to_csv('data/test monthly counts county panel season-adj.csv')
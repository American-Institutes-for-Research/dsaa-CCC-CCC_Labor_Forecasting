import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
from dateutil.relativedelta import relativedelta
county_names = ['Cook, IL', 'DuPage, IL', 'Lake, IL', 'Will, IL', 'Kane, IL',
       'Lake, IN', 'McHenry, IL', 'Kenosha, WI', 'Porter, IN', 'DeKalb, IL',
       'Kendall, IL', 'Grundy, IL', 'Jasper, IN', 'Newton, IN']

for c in county_names:
    # check for seasonality.
    df = pd.read_csv('data/test monthly counts '+c+'.csv')
    df = df.rename({'Unnamed: 0':'date'}, axis=1)
    df['month']= df['date'].str[5:7].astype('int')
    df = df.fillna(method='ffill')
    # 7-55 filter is to remove months with 0 obs
    df = df.iloc[7:55,:].reset_index(drop=True)

    # create times series index
    date_idx = pd.to_datetime(df['date'])

    # normalize all columns based on job postings counts
    df = df.drop('date', axis=1)
    job_counts = df['Postings count'].copy()
    raw_df = df.copy()
    df = df.divide(job_counts, axis=0)
    df['Postings count'] = job_counts
    df = df.set_index(pd.DatetimeIndex(date_idx))
    targets = raw_df.mean(numeric_only=True).loc[raw_df.mean(numeric_only=True)>100].index

    ccc_df = pd.read_excel('emsi_skills_api/course_skill_counts.xlsx')
    ccc_df.columns = ['skill', 'count']
    ccc_skills = ['Skill: ' + i for i in ccc_df['skill']]
    targets = set(ccc_skills).intersection(set(targets)).union(set(['Postings count']))
    targets = list(targets)
    targets.sort()

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
        if t != 'Postings count':
            result = seasonal_decompose(df[t])
            clean_df[t] = result.trend

    clean_df = clean_df[targets]

    clean_df = clean_df.dropna()
    clean_df.to_csv('data/county/test monthly counts season-adj '+c+'.csv')
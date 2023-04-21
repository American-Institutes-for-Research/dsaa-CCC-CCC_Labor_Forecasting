# create data sets without seasonality adjustment

import pandas as pd

def prepare_data(df, filename = None):
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
    if filename is not None:
        df.to_csv(filename)
    return df

df = pd.read_csv('data/test monthly counts.csv')
df = prepare_data(df, filename= 'data/test monthly counts non-season-adj skill.csv')
df2 = pd.read_csv('data/test monthly counts categories.csv')
cat_df = df2[[i for i in df2.columns if 'Skill cat:' in i] + ['Unnamed: 0', 'Postings count']]
cat_df = prepare_data(cat_df, filename= 'data/test monthly counts non-season-adj category.csv')
scat_df = df2[[i for i in df2.columns if 'Skill subcat:' in i] + ['Unnamed: 0', 'Postings count']]
scat_df = prepare_data(scat_df, filename= 'data/test monthly counts non-season-adj subcategory.csv')

# repeat for all of the counties
county_names = ['Cook, IL', 'DuPage, IL', 'Lake, IL', 'Will, IL', 'Kane, IL',
       'Lake, IN', 'McHenry, IL', 'Kenosha, WI', 'Porter, IN', 'DeKalb, IL',
       'Kendall, IL', 'Grundy, IL', 'Jasper, IN', 'Newton, IN']

county_dfs = {}
c_cat_dfs = {}
c_scat_dfs = {}
for c in county_names:
    cdf = pd.read_csv('data/test monthly counts ' +c+'.csv')
    county_dfs[c] = prepare_data(cdf)
    cdf2 = pd.read_csv('data/test monthly counts '+c+' categories.csv')
    cat_df = df2[[i for i in df2.columns if 'Skill cat:' in i] + ['Unnamed: 0', 'Postings count']]
    c_cat_dfs[c] = prepare_data(cat_df)
    scat_df = df2[[i for i in df2.columns if 'Skill subcat:' in i] + ['Unnamed: 0', 'Postings count']]
    c_scat_dfs[c] = prepare_data(scat_df)

master_df = pd.DataFrame()
master_df2 = pd.DataFrame()
master_df3 = pd.DataFrame()
for c in county_names:
    df = c_cat_dfs[c]
    df2 = c_scat_dfs[c]
    df3 = county_dfs[c]
    df['county'] = c
    df2['county'] = c
    df3['county'] = c
    df = df.set_index('county', append=True)
    df2 = df2.set_index('county', append=True)
    df3 = df3.set_index('county', append=True)
    master_df = pd.concat([master_df, df])
    master_df2 = pd.concat([master_df2, df2])
    master_df3 = pd.concat([master_df3, df3])
master_df.to_csv('data/test monthly counts non-season-adj county category.csv')
master_df2.to_csv('data/test monthly counts non-season-adj county subcategory.csv')
master_df3.to_csv('data/test monthly counts non-season-adj county skill.csv')

# create features for each skill based on average stat changes in QCEW data for NAICS codes within skill

import pandas as pd
import os


qcew_df = pd.read_excel('QCEW/QCEW_Cook_2015_2021.xlsx', index_col= 0)
temp_df = qcew_df['yr.qtr.month'].str.split('.',expand = True)
temp_df.columns = ['yr','qtr','month']
temp_df['month'] = (temp_df['qtr'].astype('int') - 1) * 3 + temp_df['month'].astype('int')
qcew_df = qcew_df.merge(temp_df, left_index= True, right_index=True)

# create data set that's lagged
qcew_df = qcew_df[['naics','ownership','yr','qtr','month','emp_month']]
qcew_df = qcew_df.sort_values(['naics','ownership','yr','qtr','month'])
qcew_df['emp_12mo_lag'] = qcew_df.groupby(['naics','ownership'])['emp_month'].shift(12)
qcew_df['emp_12mo_chg'] = 1- (qcew_df['emp_12mo_lag']/qcew_df['emp_month'])
# limit to 2018 and later

filenum = len(os.listdir('../data/us_postings'))
my_list = os.listdir('../data/us_postings')

# load job postings files, attach QCEW data based on NAICS code/month,
for i, f in enumerate(my_list):
    print('chunk', i, 'of', len(my_list), '-', f)
    df = pd.read_csv('data/us_postings/' + f)

    df['POSTED'] = pd.to_datetime(df.EXPIRED).dt.date
    df = df.loc[df.POSTED.isna() == False]
    df['POSTED_MONTH'] = pd.to_datetime(df.POSTED).dt.month
    df['POSTED_MONTH'] = df['POSTED_MONTH'].astype('int')
    df['POSTED_YEAR'] = pd.to_datetime(df.POSTED).dt.year
    df['POSTED_YEAR'] = df['POSTED_YEAR'].astype('int')
    df['EXPIRED'] = pd.to_datetime(df.EXPIRED).dt.date
    df['EXPIRED_MONTH'] = pd.to_datetime(df.EXPIRED).dt.month
    df['EXPIRED_YEAR'] = pd.to_datetime(df.EXPIRED).dt.year
    df.loc[df.EXPIRED_MONTH.isna() == False, 'EXPIRED_MONTH'] = df.loc[
        df.EXPIRED_MONTH.isna() == False, 'EXPIRED_MONTH'].astype('int')
    df.loc[df.EXPIRED_YEAR.isna() == False, 'EXPIRED_YEAR'] = df.loc[
        df.EXPIRED_YEAR.isna() == False, 'EXPIRED_YEAR'].astype('int')

    # establish range of dates to be analyzed
    min_date = datetime.date(2018, 1, 1)
    max_date = datetime.date(2022, 8, 1)
    df = df.loc[df.POSTED > min_date]
    df = df.loc[df.POSTED < max_date]

    pass
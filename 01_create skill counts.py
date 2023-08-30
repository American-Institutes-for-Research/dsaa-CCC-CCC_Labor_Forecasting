'''
Luke Patterson
01_create skill counts.py

Purpose: Create a dataset that represents the number of job postings demanding each skill by month, at the skill level
of the EMSI skill hierarchy.

input: data/us_postings/*.csv.gz - raw job postings data obtained from EMSI

output: data/test monthly counts.csv

'''

import pandas as pd
import os
import datetime
import warnings

warnings.simplefilter('ignore')

# import and append all data files

# tot_df = pd.read_csv('data/test monthly counts.csv')

filenum = len(os.listdir('data/2023_update'))
# n = 10
my_list = os.listdir('data/2023_update')
# tot_df = pd.read_csv('data/test monthly counts checkpoint.csv', index_col=0)
first = True
for i,df in enumerate(pd.read_csv('data/2023_update/AIR Datapull Expanded.csv', chunksize=10000)):
    if i > -1:
        print('chunk', i)
        with open('working/01_tracker.txt', 'w') as file:
            file.write(str(i))
            file.close()

        #df = pd.read_csv('data/2023_update/' + f)

        df['POSTED'] = pd.to_datetime(df.POSTED).dt.date
        df = df.loc[df.POSTED.isna() == False]
        df['POSTED_MONTH'] = pd.to_datetime(df.POSTED).dt.month
        df['POSTED_MONTH'] = df['POSTED_MONTH'].astype('int').round(0)
        df['POSTED_YEAR'] = pd.to_datetime(df.POSTED).dt.year
        df['POSTED_YEAR'] = df['POSTED_YEAR'].astype('int').round(0)
        df['EXPIRED'] = pd.to_datetime(df.EXPIRED).dt.date
        df['EXPIRED_MONTH'] = pd.to_datetime(df.EXPIRED).dt.month.fillna(99)
        df['EXPIRED_YEAR'] = pd.to_datetime(df.EXPIRED).dt.year.fillna(9999)
        df.loc[df.EXPIRED_MONTH.isna() == False, 'EXPIRED_MONTH'] = df.loc[
            df.EXPIRED_MONTH.isna() == False, 'EXPIRED_MONTH'].astype('int').round(0)
        df.loc[df.EXPIRED_YEAR.isna() == False, 'EXPIRED_YEAR'] = df.loc[
            df.EXPIRED_YEAR.isna() == False, 'EXPIRED_YEAR'].astype('int').round(0)

        # establish range of dates to be analyzed
        min_date = datetime.date(2018, 1, 1)
        max_date = datetime.date(2023, 8, 1)
        df = df.loc[df.POSTED > min_date]
        df = df.loc[df.POSTED < max_date]

        # make df of binary indicators for skills
        skilldf = df[['ID', 'POSTED', 'EXPIRED_MONTH', "EXPIRED_YEAR", 'EXPIRED',
                      'POSTED_MONTH', "POSTED_YEAR", 'SKILLS_NAME']].copy()
        skilldf['SKILLS_NAME'] = skilldf['SKILLS_NAME'].apply(lambda x: eval(x))
        # make unique list of skills
        skills = list(set([item for sublist in skilldf['SKILLS_NAME'].values for item in sublist]))
        print('Num of Skills:', len(skills))
        for n, s in enumerate(skills):
            has_skill = skilldf.SKILLS_NAME.apply(lambda x: s in x)
            skilldf['Skill: ' + s] = 0
            skilldf.loc[has_skill, 'Skill: ' + s] = 1
        skill_cols = ['Skill: ' + s for s in skills]

        cdf = pd.DataFrame(columns=skill_cols)
        cdf['Postings count'] = 0
        skilldf['POSTED_YYYYMM'] = skilldf.POSTED_YEAR.astype('str') + skilldf.POSTED_MONTH.astype('str').str.zfill(2)
        skilldf['POSTED_YYYYMM'] = skilldf['POSTED_YYYYMM'].astype('int')
        skilldf['EXPIRED_YYYYMM'] = skilldf.EXPIRED_YEAR.astype("str") + skilldf.EXPIRED_MONTH.astype('str').str.zfill(2)
        skilldf['EXPIRED_YYYYMM'] = skilldf['EXPIRED_YYYYMM'].astype('int')

        #dates = skilldf['POSTED_YYYYMM'].unique()
        #dates.sort()
        dates = [int(str(y) + str(m).zfill(2)) for y in range(2018, 2024) for m in range(1, 13)]

        for yyyymm in dates:
            filt_df = skilldf.loc[(skilldf.POSTED_YYYYMM <= yyyymm) &
                                  ((skilldf.EXPIRED_YYYYMM.isna()) |
                                   (skilldf.EXPIRED_YYYYMM >= yyyymm))]
            str_ym = str(yyyymm)
            y = int(str_ym[0:4])
            m = int(str_ym[4:6])
            filt_sum = pd.Series(filt_df[skill_cols].sum(), name=datetime.date(y, m, 1))
            # TODO: sensitivity check for duplicated skills.
            filt_sum['Postings count'] = filt_df.shape[0]
            cdf = cdf.append(filt_sum)
        cdf.index = cdf.index.map(str)
        if first:
            tot_df = cdf
            first = False
        else:
            exist_cols = [i for i in cdf.columns if i in tot_df.columns]
            for col in exist_cols:
                result = tot_df[col].fillna(0) + cdf[col].fillna(0)
                #assert all(result >= tot_df[col].fillna(0))
                tot_df[col] = result
            new_cols = [i for i in cdf.columns if i not in tot_df.columns]
            tot_df = tot_df.merge(cdf[new_cols].fillna(0), left_index=True, right_index=True, how='left')
            tot_df = tot_df.fillna(0)
            tot_df.to_csv('data/test monthly counts 2023 update.csv')
        pass

import pandas as pd
import os
import datetime

# import and append all data files

#tot_df = pd.read_csv('data/test monthly counts.csv')

filenum = len(os.listdir('data/us_postings'))
n = 10
my_list = os.listdir('data/us_postings')
tot_df = pd.read_csv('data/test monthly counts checkpoint.csv', index_col=0)
for i, f in enumerate(my_list):
    if i > 212 and '.csv.gz' in f:
        print('chunk',i,'of', len(my_list), '-',f)
        df = pd.read_csv('data/us_postings/'+f)

        df['POSTED'] = pd.to_datetime(df.POSTED).dt.date
        df['POSTED_MONTH'] = pd.to_datetime(df.POSTED).dt.month
        df['POSTED_YEAR'] = pd.to_datetime(df.POSTED).dt.year
        df['EXPIRED'] = pd.to_datetime(df.EXPIRED).dt.date
        # establish range of dates to be analyzed
        min_date = datetime.date(2018,1,1)
        max_date = datetime.date(2022,8,1)
        df = df.loc[df.POSTED> min_date]
        df = df.loc[df.POSTED < max_date]

        # make df of binary indicators for skills
        skilldf = df[['ID','POSTED','POSTED_MONTH',"POSTED_YEAR",'EXPIRED','SKILLS_NAME']].copy()
        skilldf['SKILLS_NAME'] = skilldf['SKILLS_NAME'].apply(lambda x: eval(x))
        # make unique list of skills
        skills = list(set([item for sublist in skilldf['SKILLS_NAME'].values for item in sublist]))
        print('Num of Skills:', len(skills))
        for n, s in enumerate(skills):
            has_skill = skilldf.SKILLS_NAME.apply(lambda x: s in x)
            skilldf['Skill: ' + s] = 0
            skilldf.loc[has_skill,'Skill: ' + s] = 1
        skill_cols = ['Skill: ' + s for s in skills]

        cdf = pd.DataFrame(columns = skill_cols)
        cdf['Postings count'] = 0
        for y in range(2018, 2023):
            for m in range(1,12):
                filt_df = skilldf.loc[(skilldf.POSTED_MONTH == m) & (skilldf.POSTED_YEAR == y)]
                filt_sum = pd.Series(filt_df[skill_cols].sum(), name = datetime.date(y,m,1))
                filt_sum['Postings count'] = filt_df.shape[0]
                cdf = cdf.append(filt_sum)
        cdf.index = cdf.index.map(str)
        if i == 0:
            tot_df = cdf
        else:
            exist_cols = [i for i in cdf.columns if i in tot_df.columns]
            for col in exist_cols:
                tot_df[col] = tot_df[col] + cdf[col]
            new_cols = [i for i in cdf.columns if i not in tot_df.columns]
            tot_df = tot_df.merge(cdf[new_cols], left_index=True, right_index = True)

            tot_df.to_csv('data/test monthly counts.csv')
        pass
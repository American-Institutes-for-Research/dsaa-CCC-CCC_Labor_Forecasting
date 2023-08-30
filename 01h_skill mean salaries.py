'''
01h_skill mean salaries.py
Author:Luke Patterson
Purpose: create mean salaries of each skill
'''
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# filenum = len(os.listdir('data/us_postings'))
# # n = 10
# my_list = os.listdir('data/us_postings')
# tot_df = pd.read_csv('data/test monthly counts checkpoint.csv', index_col=0)
first = True
sal_df = pd.DataFrame()


for i,df in enumerate(pd.read_csv('data/2023_update/AIR Datapull Expanded.csv', chunksize=10000)):
    if i > -1:
        print('chunk', i)
        with open('working/01_tracker.txt', 'w') as file:
            file.write(str(i))
            file.close()

        df['SKILLS_NAME'] = df['SKILLS_NAME'].apply(lambda x: eval(x))
        df['salary'] = df[['SALARY_TO','SALARY_FROM']].mean(axis=1)
        # make unique list of skills
        skills = list(set([item for sublist in df['SKILLS_NAME'].values for item in sublist]))
        print('Num of Skills:', len(skills))

        # keep a running tally of the sum and count of salaries for each skill. Used to calculate mean at the end
        sal_tot = pd.Series()
        sal_counts = pd.Series()
        for n, s in enumerate(skills):
            filt_df = df.loc[df.SKILLS_NAME.apply(lambda x: s in x)]
            sal_tot[s] = filt_df.salary.sum()
            sal_counts[s] = len(filt_df.salary.dropna())
        if first:
            sal_df['tot'] = sal_tot
            sal_df['counts'] = sal_counts
        else:
            # combine indexes together so they can be added properly
            #comb_index = pd.Index.union(sal_df.index, pd.Index(skills))
            sal_df = pd.concat([sal_df, pd.DataFrame(index=skills)])
            sal_df = sal_df[~sal_df.index.duplicated(keep='first')]
            sal_df = sal_df.fillna(0)

            sal_tot = pd.concat([sal_tot, pd.Series(index=sal_df.index)])
            sal_tot = sal_tot[~sal_tot.index.duplicated(keep='first')]
            sal_tot = sal_tot.fillna(0)

            sal_counts = pd.concat([sal_counts, pd.Series(index=sal_df.index)])
            sal_counts = sal_counts[~sal_counts.index.duplicated(keep='first')]
            sal_counts = sal_counts.fillna(0)

            sal_df['tot'] = sal_df['tot'] + sal_tot
            sal_df['tot'] = sal_df['tot'].fillna(0)
            sal_df['counts'] = sal_df['counts'] + sal_counts
            sal_df['counts'] = sal_df['counts'].fillna(0)

        first = False

        sal_df.to_csv('output/skill_salaries.csv')

sal_df = pd.read_csv('output/skill_salaries.csv', index_col=0)
sal_df['mean'] =sal_df.tot / sal_df.counts
sal_df.to_csv('output/skill_salaries with means.csv')

pass

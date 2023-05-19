'''
01f_create ind_skill matrix.py
Author:Luke Patterson
Purpose: create a matrix of the relative frequency between skills demanded and industrys of job postings
'''

import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

filenum = len(os.listdir('data/us_postings'))
# n = 10
my_list = os.listdir('data/us_postings')
# tot_df = pd.read_csv('data/test monthly counts checkpoint.csv', index_col=0)
first = True
ind_df = pd.DataFrame()
for i, f in enumerate(my_list):
    if i > -1 and '.csv.gz' in f:
        print('chunk', i, 'of', len(my_list), '-', f)
        with open('working/01_tracker.txt', 'w') as file:
            file.write(str(i))
            file.close()

        df = pd.read_csv('data/us_postings/' + f)
        df['SKILLS_NAME'] = df['SKILLS_NAME'].apply(lambda x: eval(x))
        # make unique list of skills
        skills = list(set([item for sublist in df['SKILLS_NAME'].values for item in sublist]))
        print('Num of Skills:', len(skills))
        ind_index = pd.MultiIndex.from_frame(df[['NAICS3','NAICS3_NAME']].drop_duplicates())
        if first:
            ind_df = pd.DataFrame(index = ind_index)
        else:
            comb_index = pd.Index.union(ind_df.index, ind_index)
            ind_df = pd.concat([ind_df,pd.DataFrame(index=comb_index)])
            ind_df = ind_df[~ind_df.index.duplicated(keep='first')]

        for n, s in enumerate(skills):
            filt_df = df.loc[df.SKILLS_NAME.apply(lambda x: s in x)]
            ind_counts = filt_df.groupby(['NAICS3','NAICS3_NAME']).count()['ID']
            if s in ind_df.columns:
                ind_df[s] = ind_df[s].add(ind_counts, fill_value = 0)
            else:
                ind_df[s] = ind_counts

            #print(ind_df)
            pass
        if first:
            ind_df['total postings'] = df.groupby(['NAICS3','NAICS3_NAME']).count()['ID']
            first = False

        else:
            ind_df['total postings'] = ind_df['total postings'].add(df.groupby(['NAICS3','NAICS3_NAME']).count()['ID'], fill_value = 0)

        ind_df.to_csv('output/skill_industry matrix_3dig.csv')

# create normalized percentages of industry matrix
#ind_df = pd.read_csv('output/skill_industry matrix_3dig.csv')
#ind_df = ind_df.set_index(['NAICS3','NAICS3_NAME'], drop=True)
ind_df = ind_df.fillna(0)

ind_df_rows = ind_df.div(ind_df.sum(axis=1), axis=0)
ind_df_cols = ind_df.div(ind_df.sum(axis=0), axis=1)

# for every skill, name top 5 industrys that demand the skill most frequently
result_df = pd.DataFrame()
for n, skill in enumerate(ind_df.columns):
    if n % 100 == 0:
        print(n)
    skill_count = ind_df_cols[skill].sort_values(ascending=False)
    row = pd.Series([i[1] for i in list(skill_count.head(5).index)], name= skill)
    result_df = pd.concat([result_df, row], axis=1)

result_df = result_df.T
result_df.to_csv('output/most common 3dig industries for each skill.csv')

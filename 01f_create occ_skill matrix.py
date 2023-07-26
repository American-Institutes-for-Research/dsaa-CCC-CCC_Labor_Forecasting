'''
01f_create occ_skill matrix.py
Author:Luke Patterson
Purpose: create a matrix of the relative frequency between skills demanded and occupations of job postings
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
occ_df = pd.DataFrame()
# for i, f in enumerate(my_list):
#     if i > -1 and '.csv.gz' in f:
#         print('chunk', i, 'of', len(my_list), '-', f)
#         with open('working/01_tracker.txt', 'w') as file:
#             file.write(str(i))
#             file.close()
#
#         df = pd.read_csv('data/us_postings/' + f)
#         df['SKILLS_NAME'] = df['SKILLS_NAME'].apply(lambda x: eval(x))
#         # make unique list of skills
#         skills = list(set([item for sublist in df['SKILLS_NAME'].values for item in sublist]))
#         print('Num of Skills:', len(skills))
#         occ_index = pd.MultiIndex.from_frame(df[['SOC_2021_3','SOC_2021_3_NAME']].drop_duplicates())
#         if first:
#             occ_df = pd.DataFrame(index = occ_index)
#         else:
#             comb_index = pd.Index.union(occ_df.index, occ_index)
#             occ_df = pd.concat([occ_df,pd.DataFrame(index=comb_index)])
#             occ_df = occ_df[~occ_df.index.duplicated(keep='first')]
#
#         for n, s in enumerate(skills):
#             filt_df = df.loc[df.SKILLS_NAME.apply(lambda x: s in x)]
#             occ_counts = filt_df.groupby(['SOC_2021_3','SOC_2021_3_NAME']).count()['ID']
#             if s in occ_df.columns:
#                 occ_df[s] = occ_df[s].add(occ_counts, fill_value = 0)
#             else:
#                 occ_df[s] = occ_counts
#
#             #print(occ_df)
#             pass
#         if first:
#             occ_df['total postings'] = df.groupby(['SOC_2021_3','SOC_2021_3_NAME']).count()['ID']
#             first = False
#
#         else:
#             occ_df['total postings'] = occ_df['total postings'].add(df.groupby(['SOC_2021_3','SOC_2021_3_NAME']).count()['ID'], fill_value = 0)
#
#         occ_df.to_csv('output/skill_occupation 3dig matrix.csv')

# create normalized percentages of occupation matrix
occ_df = pd.read_csv('output/skill_occupation 3dig matrix.csv')
occ_df = occ_df.set_index(['SOC_2021_3','SOC_2021_3_NAME'], drop=True)
occ_df = occ_df.fillna(0)

occ_df_rows = occ_df.div(occ_df.sum(axis=1), axis=0)
occ_df_cols = occ_df.div(occ_df.sum(axis=0), axis=1)

# for every skill, name top 5 occupations that demand the skill most frequently
result_df = pd.DataFrame()
for n, skill in enumerate(occ_df.columns):
    if n % 100 == 0:
        print(n)
    skill_count = occ_df_cols[skill].sort_values(ascending=False)
    row = pd.Series([i[1] for i in list(skill_count.head(5).index)], name= skill)
    result_df = pd.concat([result_df, row], axis=1)

result_df = result_df.T
result_df.to_csv('output/most common 3dig occupations for each skill.csv')

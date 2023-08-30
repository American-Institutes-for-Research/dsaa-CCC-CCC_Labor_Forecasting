'''
Luke Patterson
01_create skill counts COE occs.py

Purpose: Create a dataset that represents the number of job postings demanding each skill by month, at the skill level
of the EMSI skill hierarchy, for each of the Occs in the COE

input: data/us_postings/*.csv.gz - raw job postings data obtained from EMSI

output: data/test monthly counts.csv

'''

import pandas as pd
import os
import datetime
import warnings

basepath = 'C:/AnacondaProjects/CCC forecasting'
os.chdir(basepath)
warnings.simplefilter(action='ignore')

# import and append all data files

# tot_df = pd.read_csv('data/test monthly counts.csv')

# get list of COEs and associated occupations
coe_df = pd.read_csv('emsi_skills_api/SOCS by COE.csv', index_col=0)
coe_occs = coe_df.to_dict()
coes = list(coe_occs.keys())
for coe in coes:
    coe_occs[coe] = list(coe_occs[coe].values())

# create dictionaries of all created categories and the skills within the categories
cat_df = pd.read_excel('emsi_skills_api/EMSI_skills_with_categories.xlsx')
cats = cat_df['category_clean'].dropna().unique()
scats = cat_df['subcategory_clean'].dropna().unique()
cats_dict = {key:list() for key in cats}
scats_dict = {key:list() for key in scats}
for label, row in cat_df.iterrows():
    if not pd.isna(row.category_clean):
        cats_dict[row['category_clean']].append(row['name'])
    if not pd.isna(row.subcategory_clean):
        scats_dict[row['subcategory_clean']].append(row['name'])

filenum = len(os.listdir('data/2023_update'))
my_list = os.listdir('data/2023_update')
first = True
tot_dfs = {i:None for i in coes}

for idx, df in enumerate(pd.read_csv('data/2023_update/AIR Datapull Expanded.csv', chunksize=10000)):
    if idx > -1:
        with open('working/01c_tracker.txt', 'w') as file:
            file.write(str(idx))
            file.close()

        print('chunk', idx)
        with open('working/01c_tracker.txt', 'w') as file:
            file.write(str(idx))
            file.close()

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
        max_date = datetime.date(2022, 8, 1)
        df = df.loc[df.POSTED > min_date]
        df = df.loc[df.POSTED < max_date]

        # make df of binary indicators for skill categories
        skilldf = df[['ID','COUNTY_NAME', 'POSTED', 'EXPIRED_MONTH', "EXPIRED_YEAR", 'EXPIRED',
                      'POSTED_MONTH', "POSTED_YEAR", 'SKILLS_NAME','SOC_2021_5']].copy()
        skilldf['SKILLS_NAME'] = skilldf['SKILLS_NAME'].apply(lambda x: eval(x))

        # clean up dates
        skilldf['POSTED_YYYYMM'] = skilldf.POSTED_YEAR.astype('str') + skilldf.POSTED_MONTH.astype('str').str.zfill(2)
        skilldf['POSTED_YYYYMM'] = skilldf['POSTED_YYYYMM'].astype('int')
        skilldf['EXPIRED_YYYYMM'] = skilldf.EXPIRED_YEAR.astype("str") + skilldf.EXPIRED_MONTH.astype('str').str.zfill(2)
        skilldf['EXPIRED_YYYYMM'] = skilldf['EXPIRED_YYYYMM'].astype('int').round(0)

        # flag which COEs the posting's occupation falls into
        for coe in coes:
            occs = coe_occs[coe]
            has_coe = skilldf.SOC_2021_5.apply(lambda x: x in occs)
            skilldf['COE flag: ' + coe] = 0
            skilldf.loc[has_coe,'COE flag: ' + coe] = 1
        # for each category, flag whether a skill from the category is present
        for n, c in enumerate(cats):
            # for each row, flag whether any skills in the list in SKILLS_NAME column match skills in skill categories
            has_cat = skilldf.SKILLS_NAME.apply(lambda skills: any([i in skills for i in cats_dict[c]]))
            skilldf['Skill cat: ' + c] = 0
            skilldf.loc[has_cat, 'Skill cat: ' + c] = 1

        # repeat for subcategories
        for n, s in enumerate(scats):
            # for each row, flag whether any skills in the list in SKILLS_NAME column match skills in skill categories
            has_scat = skilldf.SKILLS_NAME.apply(lambda skills: any([i in skills for i in scats_dict[s]]))
            skilldf['Skill subcat: ' + s] = 0
            skilldf.loc[has_scat, 'Skill subcat: ' + s] = 1

        skills = list(set([item for sublist in skilldf['SKILLS_NAME'].values for item in sublist]))
        print('Num of Skills:', len(skills))
        for n, s in enumerate(skills):
            has_skill = skilldf.SKILLS_NAME.apply(lambda x: s in x)
            skilldf['Skill: ' + s] = 0
            skilldf.loc[has_skill, 'Skill: ' + s] = 1
        skill_cat_cols = ['Skill cat: ' + i for i in cats] + ['Skill subcat: ' + i for i in scats] + ['Skill: ' + s for s in skills]
        # for each county
        for coe in coes:
            print('counting postings for', coe)
            coe_flag = 'COE flag: ' + coe
            coe_skilldf = skilldf.loc[skilldf[coe_flag] == 1]

            # create a dataframe that will count the number of postings in each month
            coe_cdf = pd.DataFrame(columns=skill_cat_cols)
            coe_cdf['Postings count'] = 0

            # for each month from 2018-2022...
            dates = [int(str(y) + str(m).zfill(2)) for y in range(2018, 2023) for m in range(1, 13)]
            for yyyymm in dates:
                # ...filter to only the postings from that month
                filt_df = coe_skilldf.loc[(coe_skilldf.POSTED_YYYYMM <= yyyymm) &
                                      ((coe_skilldf.EXPIRED_YYYYMM.isna()) |
                                       (coe_skilldf.EXPIRED_YYYYMM >= yyyymm))]
                str_ym = str(yyyymm)
                y = int(str_ym[0:4])
                m = int(str_ym[4:6])

                # create counts of number of postings requiring skills from the category that month
                filt_sum = pd.Series(filt_df[skill_cat_cols].sum(), name=datetime.date(y, m, 1))
                filt_sum['Postings count'] = filt_df.shape[0]
                coe_cdf = coe_cdf.append(filt_sum)
            coe_cdf.index = coe_cdf.index.map(str)
            if tot_dfs[coe] is None:
                tot_dfs[coe] = coe_cdf
            else:
                exist_cols = [i for i in coe_cdf.columns if i in tot_dfs[coe].columns]
                for col in exist_cols:
                    result = tot_dfs[coe][col].fillna(0) + coe_cdf[col].fillna(0)
                    #assert all(result >= tot_df[col].fillna(0))
                    tot_dfs[coe][col] = result
                new_cols = [i for i in coe_cdf.columns if i not in tot_dfs[coe].columns]
                tot_dfs[coe] = tot_dfs[coe].merge(coe_cdf[new_cols].fillna(0), left_index=True, right_index=True, how='left')
                tot_dfs[coe] = tot_dfs[coe].fillna(0)
                tot_dfs[coe].to_csv('data/COE/test monthly counts '+ coe + ' categories.csv')
            pass

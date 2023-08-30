'''
Luke Patterson
01a_create panel data skill category.py

Purpose: Create a dataset that represents the number of job postings demanding each skill by month and county, at the
category/subcategory level of the EMSI skill hierarchy.

input: data/us_postings/*.csv.gz - raw job postings data obtained from EMSI

output: data/test monthly counts [county name] categories.csv - one file for each of the 13 counties in the Chicago MSA
'''

import os
import warnings
warnings.simplefilter(action='ignore')
import pandas as pd
import datetime

filenum = len(os.listdir('data/us_postings'))
my_list = os.listdir('data/us_postings')
# tot_df = pd.read_csv('data/test monthly counts checkpoint.csv', index_col=0)
cat_df = pd.read_excel('emsi_skills_api/EMSI_skills_with_categories.xlsx')
cats = cat_df['category_clean'].dropna().unique()
scats = cat_df['subcategory_clean'].dropna().unique()
cats_dict = {key:list() for key in cats}
scats_dict = {key:list() for key in scats}
county_names = ['Cook, IL', 'DuPage, IL', 'Lake, IL', 'Will, IL', 'Kane, IL',
       'Lake, IN', 'McHenry, IL', 'Kenosha, WI', 'Porter, IN', 'DeKalb, IL',
       'Kendall, IL', 'Grundy, IL', 'Jasper, IN', 'Newton, IN']

# create dictionaries of all created categories and the skills within the categories
tot_dfs = {i:None for i in county_names}
for label, row in cat_df.iterrows():
    if not pd.isna(row.category_clean):
        cats_dict[row['category_clean']].append(row['name'])
    if not pd.isna(row.subcategory_clean):
        scats_dict[row['subcategory_clean']].append(row['name'])

for idx, f in enumerate(my_list):
    if idx > -1 and '.csv.gz' in f:
        with open('working/01c_tracker.txt', 'w') as file:
            file.write(str(idx))
            file.close()

        print('chunk', idx, 'of', len(my_list), '-', f)
        with open('working/01c_tracker.txt', 'w') as file:
            file.write(str(idx))
            file.close()
        df = pd.read_csv('data/us_postings/' + f)

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
                      'POSTED_MONTH', "POSTED_YEAR", 'SKILLS_NAME']].copy()
        skilldf['SKILLS_NAME'] = skilldf['SKILLS_NAME'].apply(lambda x: eval(x))

        # clean up dates
        skilldf['POSTED_YYYYMM'] = skilldf.POSTED_YEAR.astype('str') + skilldf.POSTED_MONTH.astype('str').str.zfill(2)
        skilldf['POSTED_YYYYMM'] = skilldf['POSTED_YYYYMM'].astype('int')
        skilldf['EXPIRED_YYYYMM'] = skilldf.EXPIRED_YEAR.astype("str") + skilldf.EXPIRED_MONTH.astype('str').str.zfill(2)
        skilldf['EXPIRED_YYYYMM'] = skilldf['EXPIRED_YYYYMM'].astype('int').round(0)

        # for each category
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
            skill_cat_cols = ['Skill cat: ' + i for i in cats] + ['Skill subcat: ' + i for i in scats]
        # for each county
        for county in county_names:
            print('counting postings for', county)
            county_skilldf = skilldf.loc[skilldf.COUNTY_NAME == county]

            # create a dataframe that will count the number of postings in each month
            county_cdf = pd.DataFrame(columns=skill_cat_cols)
            county_cdf['Postings count'] = 0

            # for each month from 2018-2022...
            dates = [int(str(y) + str(m).zfill(2)) for y in range(2018, 2023) for m in range(1, 13)]
            for yyyymm in dates:
                # ...filter to only the postings from that month
                filt_df = county_skilldf.loc[(county_skilldf.POSTED_YYYYMM <= yyyymm) &
                                      ((county_skilldf.EXPIRED_YYYYMM.isna()) |
                                       (county_skilldf.EXPIRED_YYYYMM >= yyyymm))]
                str_ym = str(yyyymm)
                y = int(str_ym[0:4])
                m = int(str_ym[4:6])

                # create counts of number of postings requiring skills from the category that month
                filt_sum = pd.Series(filt_df[skill_cat_cols].sum(), name=datetime.date(y, m, 1))
                filt_sum['Postings count'] = filt_df.shape[0]
                county_cdf = county_cdf.append(filt_sum)
            county_cdf.index = county_cdf.index.map(str)
            if tot_dfs[county] is None:
                tot_dfs[county] = county_cdf
            else:
                exist_cols = [i for i in county_cdf.columns if i in tot_dfs[county].columns]
                for col in exist_cols:
                    result = tot_dfs[county][col].fillna(0) + county_cdf[col].fillna(0)
                    #assert all(result >= tot_df[col].fillna(0))
                    tot_dfs[county][col] = result
                new_cols = [i for i in county_cdf.columns if i not in tot_dfs[county].columns]
                tot_dfs[county] = tot_dfs[county].merge(county_cdf[new_cols].fillna(0), left_index=True, right_index=True, how='left')
                tot_dfs[county] = tot_dfs[county].fillna(0)
                tot_dfs[county].to_csv('data/test monthly counts '+ county + ' categories.csv')
            pass


# import pandas as pd
# df = pd.read_csv('data/test monthly counts.csv',index_col=0)
#
# postings_count = df['Postings count'].copy()
#
# emsi_df = pd.read_excel('emsi_skills_api/EMSI_skills_with_categories.xlsx', index_col=0)
#
# df = df.T
# df = df.reset_index().rename({'index':'skill'}, axis=1)
# df['skill'] = df['skill'].str.replace('Skill:','').str.strip()
# df = df.merge(emsi_df[['name','category_clean','subcategory_clean']], how='left', left_on='skill', right_on='name')
# df = df.drop('name', axis=1).rename({'category_clean':'category','subcategory_clean':'subcategory'}, axis=1)
#
# cat_df = df.groupby('category').sum()
# scat_df = df.groupby('subcategory').sum()
#
# cat_df = cat_df.T
# cat_df.columns = ['Skill cat: ' + i for i in cat_df.columns]
# scat_df = scat_df.T
# scat_df.columns = ['Skill subcat: ' + i for i in scat_df.columns]
#
#
# cat_df['Postings count'] = postings_count
# scat_df['Postings count'] = postings_count
#
# cat_df.to_csv('data/test monthly counts skill category.csv')
# scat_df.to_csv('data/test monthly counts skill subcategory.csv')
#
# pass

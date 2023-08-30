'''
01i_create_cat_subcat characteristics.py
Create occupation, industry, and salary information at the category and subcategory levels based on the skill level info
'''

import pandas as pd

occ_df = pd.read_csv('output/skill_occupation 3dig matrix.csv')
occ_df = occ_df.set_index(['SOC_2021_3','SOC_2021_3_NAME'], drop=True)

ind_df = pd.read_csv('output/skill_industry matrix_3dig.csv')
ind_df = ind_df.set_index(['NAICS3','NAICS3_NAME'], drop=True)

sal_df = pd.read_csv('output/skill_salaries with means.csv', index_col = 0)

cat_df = pd.read_excel('emsi_skills_api/EMSI_skills_with_categories.xlsx')

sal_df = sal_df.merge(cat_df[['name','subcategory_clean','category_clean']], left_index= True, right_on = 'name', how='left')
scat_sal_df = sal_df.groupby('subcategory_clean').sum()
scat_sal_df['mean'] =scat_sal_df.tot / scat_sal_df.counts
cat_sal_df = sal_df.groupby('category_clean').sum()
cat_sal_df['mean'] =cat_sal_df.tot / cat_sal_df.counts

cat_sal_df.to_csv('output/category_salaries with means.csv')
scat_sal_df.to_csv('output/subcategory_salaries with means.csv')

ind_df = ind_df.T.fillna(0)
occ_df = occ_df.T.fillna(0)
ind_df = ind_df.merge(cat_df[['name','subcategory_clean','category_clean']], left_index= True, right_on = 'name', how='left')
occ_df = occ_df.merge(cat_df[['name','subcategory_clean','category_clean']], left_index= True, right_on = 'name', how='left')



# subcategory and industry
scat_ind_df = ind_df.groupby('subcategory_clean').sum().T
ind_df_rows = scat_ind_df.div(scat_ind_df.sum(axis=1), axis=0)
ind_df_cols = scat_ind_df.div(scat_ind_df.sum(axis=0), axis=1)
# for every subcat, name top 5 industries that demand the subcat most frequently appears in
result_df = pd.DataFrame()
for n, skill in enumerate(scat_ind_df.columns):
    if n % 100 == 0:
        print(n)
    skill_count = ind_df_cols[skill].sort_values(ascending=False)
    row = pd.Series([i[1] for i in list(skill_count.head(5).index)], name= skill)
    result_df = pd.concat([result_df, row], axis=1)

result_df = result_df.T
result_df.to_csv('output/most common 3dig industries for each subcategory.csv')


# subcategory and occupation
scat_occ_df = occ_df.groupby('subcategory_clean').sum().T
occ_df_rows = scat_occ_df.div(scat_occ_df.sum(axis=1), axis=0)
occ_df_cols = scat_occ_df.div(scat_occ_df.sum(axis=0), axis=1)
# for every subcat, name top 5 occupations that demand the subcat most frequently
result_df = pd.DataFrame()
result_df_codes = pd.DataFrame()
for n, skill in enumerate(scat_occ_df.columns):
    if n % 100 == 0:
        print(n)
    skill_count = occ_df_cols[skill].sort_values(ascending=False)
    row = pd.Series([i[1] for i in list(skill_count.head(5).index)], name= skill)
    result_df = pd.concat([result_df, row], axis=1)
    code_row = pd.Series([i[0] for i in list(skill_count.head(5).index)], name= skill)
    result_df_codes = pd.concat([result_df_codes, code_row], axis=1)

result_df = result_df.T
result_df.to_csv('output/most common 3dig occupations for each subcategory.csv')
result_df_codes = result_df_codes.T
result_df_codes.to_csv('output/most common 3dig occupation codes for each subcategory.csv')

# category and industry
cat_ind_df = ind_df.groupby('category_clean').sum().T
ind_df_rows = cat_ind_df.div(cat_ind_df.sum(axis=1), axis=0)
ind_df_cols = cat_ind_df.div(cat_ind_df.sum(axis=0), axis=1)
# for every category, name top 5 industries that demand the category most frequently uses
result_df = pd.DataFrame()
for n, skill in enumerate(cat_ind_df.columns):
    if n % 100 == 0:
        print(n)
    skill_count = ind_df_cols[skill].sort_values(ascending=False)
    row = pd.Series([i[1] for i in list(skill_count.head(5).index)], name= skill)
    result_df = pd.concat([result_df, row], axis=1)

result_df = result_df.T
result_df.to_csv('output/most common 3dig industries for each category.csv')

# category and occupation
cat_occ_df = occ_df.groupby('category_clean').sum().T
occ_df_rows = cat_occ_df.div(cat_occ_df.sum(axis=1), axis=0)
occ_df_cols = cat_occ_df.div(cat_occ_df.sum(axis=0), axis=1)
# for every category, name top 5 occupations that demand the category most frequently uses
result_df = pd.DataFrame()
result_df_codes = pd.DataFrame()
for n, skill in enumerate(cat_occ_df.columns):
    if n % 100 == 0:
        print(n)
    skill_count = occ_df_cols[skill].sort_values(ascending=False)
    row = pd.Series([i[1] for i in list(skill_count.head(5).index)], name= skill)
    code_row = pd.Series([i[0] for i in list(skill_count.head(5).index)], name= skill)
    result_df = pd.concat([result_df, row], axis=1)
    result_df_codes = pd.concat([result_df_codes, code_row], axis=1)

result_df = result_df.T
result_df.to_csv('output/most common 3dig occupations for each category.csv')
result_df_codes = result_df_codes.T
result_df_codes.to_csv('output/most common 3dig occupation codes for each category.csv')

pass

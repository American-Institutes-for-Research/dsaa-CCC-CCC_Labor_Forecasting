'''
05_create output excel file.py
Luke Patterson, 4-15-2023
Purpose:
'''

import pandas as pd
import numpy as np
import xlsxwriter

scat_title = 'VAR_ARIMA ensemble multiple models subcategory level'
cat_title = 'VAR_ARIMA ensemble multiple models category level'
title = 'VAR_ARIMA ensemble multiple models skill level'
min_obs = 50
model_labels = ['VAR', 'ARIMA']
output_label = 'VAR_ARIMA ensemble multiple models formatted results'

# load ensemble model results for all three hierarchy levels
cat_ensemble_df = pd.read_csv('output/predicted changes/ensemble results '+cat_title+'.csv', index_col=0)
cat_ensemble_df.index = [i.replace('Skill cat: ','') for i in cat_ensemble_df.index]
scat_ensemble_df = pd.read_csv('output/predicted changes/ensemble results '+scat_title+'.csv', index_col=0)
scat_ensemble_df.index = [i.replace('Skill subcat: ','') for i in scat_ensemble_df.index]
ensemble_df = pd.read_csv('output/predicted changes/ensemble results '+title+'.csv', index_col=0)
ensemble_df.index = [i.replace('Skill: ','') for i in ensemble_df.index]

# load data frame of all skills/cats/subcats in the emsi taxonomy
cat_df = pd.read_excel('emsi_skills_api/EMSI_skills_with_categories.xlsx')
# skills = cat_df['name'].dropna().unique()
# cats = cat_df['category_clean'].dropna().unique()
# scats = cat_df['subcategory_clean'].dropna().unique()
# cats_dict = {key:list() for key in cats}
# scats_dict = {key:list() for key in scats}
# # create dictionaries of all created categories and the skills within the
# for label, row in cat_df.iterrows():
#     # if not pd.isna(row.category_clean):
#     #     cats_dict[row['category_clean']].append(row['name'])
#     if not pd.isna(row.subcategory_clean):
#         scats_dict[row['subcategory_clean']].append(row['name'])
#     if not pd.isna(row.category_clean) and not pd.isna(row.subcategory_clean) and row['subcategory_clean'] not in \
#         cats_dict[row['category_clean']]:
#         cats_dict[row['category_clean']].append(row['subcategory_clean'])

# create multi indexed dataframe with each unique value of all three levels of the skills hierarchy
pred_df = cat_df[['category_clean','subcategory_clean','name']].rename(
    {
        'category_clean':'category',
        'subcategory_clean':'subcategory',
        'name':'skill'
    }
    ,axis=1
)

# merge in skill-level predictions
pred_df = pred_df.merge(ensemble_df, left_on = 'skill', right_index= True)

# add rows for the subcategory predictions
scat_ensemble_df = scat_ensemble_df.reset_index().rename({'index':'subcategory'}, axis=1)
# add categories of subcategories
subcat_xwalk = cat_df[['category_clean','subcategory_clean']].drop_duplicates().rename({'category_clean':'category',
                                                                                        'subcategory_clean':'subcategory'}, axis=1)
scat_ensemble_df = scat_ensemble_df.merge(subcat_xwalk, left_on = 'subcategory', right_on = 'subcategory')

pred_df = pd.concat([scat_ensemble_df, pred_df])

pred_df = pred_df.sort_values(['category','subcategory','skill'])

# do some tricky sort things to get the subcategory predictions to appear at the top of their groups
pred_df = pred_df.reset_index()
pred_df = pred_df.groupby(['category', 'subcategory'], group_keys=False).apply(lambda x: x.sort_values(['index']))


# add rows for category predictions
cat_ensemble_df = cat_ensemble_df.reset_index().rename({'index':'category'}, axis=1)

# add category predictions and sort to make sure they appear at the top of groups
cat_ensemble_df = cat_ensemble_df.reset_index()
pred_df = pred_df.drop('index',axis=1).reset_index(drop=True).reset_index()

pred_df['index'] = pred_df['index'] + cat_ensemble_df.shape[0]
pred_df = pd.concat([cat_ensemble_df,pred_df])
pred_df = pred_df[['index','category','subcategory', 'skill', 'July 2022 actual', 'July 2024 predicted',
       'Percent change', 'Monthly average obs', 'model','Normalized RMSE']]

pred_df = pred_df.rename({'index':'scat_index'}, axis=1)
pred_df = pred_df.groupby(['category'], group_keys=True).apply(lambda x: x.sort_values(['scat_index']))

pred_df = pred_df.drop('scat_index')
pred_df = pred_df.loc[pred_df['Monthly average obs'] > min_obs]

# keep the name of the model embedded in the model name
pred_df['model'] = pred_df.model.apply(lambda x: [i for i in x.split(' ') if i in model_labels][0])
pred_df = pred_df.drop('scat_index',axis=1).reset_index(drop=True).reset_index().rename({'index':'rownum'}, axis=1)
pred_df['rownum'] = pred_df.rownum + 2

# identify row numbers of groups for categories and subcategories
# cat_groups = []
# scat_groups = []
# for cat in pred_df.category.unique():
#     cat_groups.append(pred_df.loc[(pred_df.category == cat)].rownum.values)
#
# for scat in pred_df.subcategory.unique():
#     scat_groups.append(pred_df.loc[pred_df.subcategory == scat].rownum.values)

# turns out excel groupings need to have summary rows for category and subcategory at the bottom of the group, so we
# will resort accordingly
pred_df = pred_df.sort_values(['category','subcategory','skill']).drop('rownum',axis=1)

writer = pd.ExcelWriter('output/exhibits/'+output_label+'.xlsx', engine='xlsxwriter')
workbook  = writer.book

pred_df.to_excel(writer, sheet_name='Grouped Skills', index= False)
worksheet = writer.sheets['Grouped Skills']

# add groupings
for n, row in pred_df.iterrows():
    if pd.isna(row.subcategory):
        worksheet.set_row(n, None, None, {'level': 0})
    elif pd.isna(row.skill):
        worksheet.set_row(n, None, None, {'level': 1, 'hidden':True})
    else:
        worksheet.set_row(n, None, None, {'level': 2, 'hidden':True})

# set summary row of each group as the top row
#worksheet.outline_settings(True, False, True, False)

# format columns to be rounded to two decimal places
format1 = workbook.add_format({'num_format': '0.000'})
worksheet.set_column('D:I', None, format1)

# autofit column widths
for column in pred_df:
    column_length = max(pred_df[column].astype(str).map(len).max(), len(column))
    col_idx = pred_df.columns.get_loc(column)
    writer.sheets['Grouped Skills'].set_column(col_idx, col_idx, column_length)

# add all predictions for each hierarchy as a separate sheet
cat_ensemble_df = cat_ensemble_df.drop('index', axis=1)
cat_ensemble_df = cat_ensemble_df.sort_values('Percent change', ascending = False)

cat_ensemble_df['model'] = cat_ensemble_df.model.apply(lambda x: [i for i in x.split(' ') if i in model_labels][0])
cat_ensemble_df.to_excel(writer, sheet_name='Category', index= False)
worksheet = writer.sheets['Category']
for column in cat_ensemble_df:
    column_length = max(cat_ensemble_df[column].astype(str).map(len).max(), len(column))
    col_idx = cat_ensemble_df.columns.get_loc(column)
    writer.sheets['Category'].set_column(col_idx, col_idx, column_length)
worksheet.set_column('B:G', None, format1)

scat_ensemble_df = scat_ensemble_df[['subcategory', 'category', 'July 2022 actual', 'July 2024 predicted',
       'Percent change', 'Monthly average obs', 'model', 'Normalized RMSE']]
scat_ensemble_df = scat_ensemble_df.sort_values('Percent change', ascending = False)
scat_ensemble_df['model'] = scat_ensemble_df.model.apply(lambda x: [i for i in x.split(' ') if i in model_labels][0])
scat_ensemble_df.to_excel(writer, sheet_name='Subcategory', index= False)
worksheet = writer.sheets['Subcategory']
for column in scat_ensemble_df:
    column_length = max(scat_ensemble_df[column].astype(str).map(len).max(), len(column))
    col_idx = scat_ensemble_df.columns.get_loc(column)
    writer.sheets['Subcategory'].set_column(col_idx, col_idx, column_length)
worksheet.set_column('B:G', None, format1)

ensemble_df = ensemble_df.merge(cat_df[['category_clean','subcategory_clean','name']], left_index=True, right_on='name')
ensemble_df = ensemble_df.rename({'category_clean':'category','subcategory_clean':'subcategory','name':'skill'},axis=1)
ensemble_df = ensemble_df[['skill','category', 'subcategory', 'July 2022 actual', 'July 2024 predicted', 'Percent change',
       'Monthly average obs', 'model', 'Normalized RMSE']]
ensemble_df = ensemble_df.sort_values('Percent change', ascending = False)
ensemble_df['model'] = ensemble_df.model.apply(lambda x: [i for i in x.split(' ') if i in model_labels][0])
ensemble_df.to_excel(writer, sheet_name='Skill', index= False)
worksheet = writer.sheets['Skill']
for column in ensemble_df:
    column_length = max(ensemble_df[column].astype(str).map(len).max(), len(column))
    col_idx = ensemble_df.columns.get_loc(column)
    writer.sheets['Skill'].set_column(col_idx, col_idx, column_length)
worksheet.set_column('B:G', None, format1)

# close the workbook
workbook.close()
pass

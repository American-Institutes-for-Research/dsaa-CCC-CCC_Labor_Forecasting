import pandas as pd
import numpy as np
dfs_2023 = pd.read_excel('output/exhibits/VAR_ARIMA ensemble overall 2023 rerun results 09262023.xlsx', sheet_name=None)
dfs_2022 = pd.read_excel('output/exhibits/VAR_ARIMA ensemble multiple models formatted results 09212023.xlsx', sheet_name=None)
coe_names = ['Business','Construction','Culinary & Hospitality','Education & Child Development','Engineering & Computer Science',
             'Health Science','Information Technology','Manufacturing','Transportation, Distribution, & Logistics']
all_dfs = {'overall_2022':dfs_2022, 'overall_2023':dfs_2023}
for coe in coe_names:
    all_dfs['COE_'+coe] = pd.read_excel('output/exhibits/VAR_ARIMA ensemble ' + coe + ' formatted results 09212023.xlsx',
                                 sheet_name=None)

lvls = ['Category','Subcategory','Skill']

# compare changes in 2022 and 2023 overall predictions

# calculate correlation between percentage/PP changes
# build multi index for the table
index = []
for lvl in lvls:
    index = index + [('actual correlation', lvl), ('predicted change correlation', lvl), ('prediction std dev correlation', lvl), ('rmse correlation', lvl)]
corr_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(index, names=['Statistic', 'Level']), columns = ['All variance levels','Low only', 'Medium only', 'High only']).sort_index()
# Define the desired sorting order for the levels and rename them
desired_order = ['Category', 'Subcategory', 'Skill']
corr_df.index = corr_df.index.set_levels(desired_order, level='Level')


for lvl in lvls:
    sub_df = dfs_2022[lvl]
    sub_df = sub_df[[lvl.lower(),'July 2022 actual', 'July 2024 weighted predicted','Percentage Point change',
                     'Percentage change','Model Variance','Prediction std dev','Average RMSE']]
    sub_df2 = dfs_2023[lvl]
    sub_df2 = sub_df2[[lvl.lower(),'July 2023 actual', 'July 2025 weighted predicted','Percentage Point change',
                       'Percentage change','Model Variance','Prediction std dev', 'Average RMSE']]
    sub_df = sub_df.rename({'Percentage Point change':'PP_change_2022','Percentage change':'P_change_2022',
                            'Model Variance':'Model Variance_2022','Prediction std dev':'Prediction std dev_2022',
                            'Average RMSE':'Average RMSE_2022'}, axis=1)
    sub_df2 = sub_df2.rename({'Percentage Point change':'PP_change_2023','Percentage change':'P_change_2023',
                              'Model Variance':'Model Variance_2023','Prediction std dev':'Prediction std dev_2023',
                              'Average RMSE':'Average RMSE_2023'}, axis=1)
    sub_df = sub_df.merge(sub_df2, on=lvl.lower(),how='outer')
    # add multi index values
    corr_df.index = corr_df.index.union([])
    corr_df.loc[('actual correlation', lvl), 'All variance levels'] = \
        sub_df[['July 2022 actual', 'July 2023 actual']].corr().min()[0]
    corr_df.loc[('predicted change correlation', lvl), 'All variance levels'] = \
        sub_df[['P_change_2022', 'P_change_2023']].corr().min()[0]
    corr_df.loc[('prediction std dev correlation', lvl), 'All variance levels'] = \
        sub_df[['Prediction std dev_2022', 'Prediction std dev_2023']].corr().min()[0]
    corr_df.loc[('rmse correlation', lvl), 'All variance levels'] = \
        sub_df[['Average RMSE_2022', 'Average RMSE_2023']].corr().min()[0]

    for var in ['Low', 'Medium', 'High']:
        # check statistics for different variance levelsskills
        filt_df = sub_df.loc[(sub_df['Model Variance_2023'] == var) &(sub_df['Model Variance_2022'] == var)]
        corr_df.loc[('predicted change correlation', lvl), var + ' only'] = \
            filt_df[['P_change_2022', 'P_change_2023']].corr().min()[0]
        corr_df.loc[('Prediction std dev correlation', lvl), var + ' only'] = filt_df[['Prediction std dev_2022', 'Prediction std dev_2023']].corr().min()[0]
        corr_df.loc[('rmse correlation', lvl), var + ' only'] = \
            filt_df[['Average RMSE_2022', 'Average RMSE_2023']].corr().min()[0]
    if lvl == 'Subcategory':
        scat_df = sub_df
    if lvl == 'Skill':
        skill_df = sub_df
corr_df.to_excel('output/comparison/correlation of 2022 and 2023 results.xlsx')

scat_df['pred diff'] = abs(scat_df['July 2025 weighted predicted'] - scat_df['July 2024 weighted predicted'])
skill_df['pred diff'] = abs(skill_df['July 2025 weighted predicted'] - skill_df['July 2024 weighted predicted'])
scat_df['act diff'] = abs(scat_df['July 2023 actual'] - scat_df['July 2022 actual'])
skill_df['act diff'] = abs(skill_df['July 2023 actual'] - skill_df['July 2022 actual'])
scat_df['pred_diff - act_diff'] = scat_df['pred diff'] - scat_df ['act diff']
skill_df['pred_diff - act_diff'] = skill_df['pred diff'] - skill_df ['act diff']

scat_df = scat_df.sort_values('pred diff', ascending = False)
skill_df = skill_df.sort_values('pred diff', ascending = False)

keep_cols = ['July 2022 actual','July 2023 actual', 'July 2024 weighted predicted', 'July 2025 weighted predicted','pred diff','act diff', 'pred_diff - act_diff']
scat_df[['subcategory'] + keep_cols].head(20).to_excel('output/comparison/top 20 changes in subcategories.xlsx')
skill_df[['skill'] + keep_cols].head(50).to_excel('output/comparison/top 50 changes in skills.xlsx')

# measure RMSE of models
rmse_df = pd.DataFrame()
# also measure RMSE for only medium/low variance models
med_rmse_df = pd.DataFrame()
# RMSE for high sample size
high_ss_rmse_df = pd.DataFrame()
# measure Prediction Std Dev to Prediction ratio
pred_dev_ratio_df = pd.DataFrame()
# number of skills/categories predictions made for
num_preds_df = pd.DataFrame()
for lvl in lvls:
    for key in all_dfs.keys():
        df = all_dfs[key][lvl]

        # some cleaning for analyses
        df = df.rename({'July 2022 actual':'Latest Actual','July 2023 actual':'Latest Actual', 'July 2024 weighted predicted':'2-Yr Prediction', 'July 2025 weighted predicted':'2-Yr Prediction'}, axis =1)
        df['std_est_ratio'] = df['Prediction std dev'] / df['2-Yr Prediction']

        # overall RMSE
        rmse_df.loc[key,lvl] = df['Average RMSE'].mean()

        # RMSE of only medium/low variance models
        filt_df = df.loc[df['Model Variance'] != 'High']
        med_rmse_df.loc[key,lvl] = filt_df['Average RMSE'].mean()

        # RMSE of high sample size skills, defined as over 1000 obs
        #top_25p = max(round(filt_df.shape[0]),1)
        # filt_df2 = filt_df.sort_values("Monthly average obs",ascending=False).head(top_25p)
        filt_df2 = filt_df.loc[filt_df['Monthly average obs']> 1000].sort_values("Monthly average obs", ascending=False)
        if filt_df2.shape[0] > 0:
            high_ss_rmse_df.loc[key,lvl] = filt_df2['Average RMSE'].mean()
        else:
            high_ss_rmse_df.loc[key, lvl] = np.nan

        # Prediction Std Dev to Prediction ratio
        pred_dev_ratio_df.loc[key,lvl] = df['std_est_ratio'].mean()

        # number of predictions made
        num_preds_df.loc[key,lvl] = df.shape[0]

rmse_df.to_excel('output/comparison/average RMSE.xlsx')
med_rmse_df.to_excel('output/comparison/average med_low RMSE.xlsx')
high_ss_rmse_df.to_excel('output/comparison/average high samp size RMSE.xlsx')
pred_dev_ratio_df.to_excel('output/comparison/average std dev ratio.xlsx')
num_preds_df.to_excel('output/comparison/num predictions made.xlsx')
pass


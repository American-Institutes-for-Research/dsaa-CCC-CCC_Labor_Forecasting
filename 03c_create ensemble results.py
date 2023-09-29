'''
03c_create ensemble results.py
Luke Patterson
04-15-2023

calls to create ensemble results between two or more different runs based on best RMSE
'''

from utils import create_ensemble_results
import os

var_runs = os.listdir('output/batch_VAR top grid search runs 2023 rerun')
arima_runs = os.listdir('output/batch_ARIMA top grid search runs 2023')
var_skill_runs = [i.replace('.csv','') for i in var_runs if 'skill' in i]
var_subcat_runs = [i.replace('.csv','') for i in var_runs if 'subcategory' in i]
var_cat_runs = [i.replace('.csv','') for i in var_runs if ' category' in i]
arima_skill_runs = [i.replace('.csv','') for i in arima_runs if 'skill' in i]
arima_subcat_runs = [i.replace('.csv','') for i in arima_runs if 'subcategory' in i]
arima_cat_runs = [i.replace('.csv','') for i in arima_runs if ' category' in i]

create_ensemble_results(
    runnames=var_cat_runs + arima_cat_runs,
    labels=['VAR' + str(i) for i in range(len(var_cat_runs))] +
           ['ARIMA' + str(i) for i in range(len(arima_cat_runs))],
    types=['VAR' for i in range(len(var_cat_runs))] +
           ['ARIMA' for i in range(len(arima_cat_runs))],
    panel_indicators=[
        False for i in range(len(var_cat_runs + arima_cat_runs))
    ],
    batch_names = ['batch_ARIMA top grid search runs 2023','batch_VAR top grid search runs 2023 rerun'],
    title='VAR_ARIMA ensemble overall 2023 rerun category level reformatted 09262023',
    hierarchy_lvl='category',
    model_selection= 'weighted average',
    output_occ_codes= True,
    rerun_2023= True,
    do_results_analysis= True
)


create_ensemble_results(
    runnames=var_subcat_runs + arima_subcat_runs,
    labels=['VAR'+str(i) for i in range(len(var_subcat_runs))] +
        ['ARIMA'+str(i) for i in range(len(arima_subcat_runs))],
    types=['VAR' for i in range(len(var_subcat_runs))] +
        ['ARIMA' for i in range(len(arima_subcat_runs))],
    panel_indicators=[
        False for i in range(len(var_subcat_runs + arima_subcat_runs))
    ],
    title='VAR_ARIMA ensemble overall 2023 rerun subcategory level reformatted 09262023',
    batch_names = ['batch_ARIMA top grid search runs 2023','batch_VAR top grid search runs 2023 rerun'],
    hierarchy_lvl='subcategory',
    model_selection='weighted average',
    output_occ_codes=True,
    rerun_2023= True,
    do_results_analysis= True
)

create_ensemble_results(
    runnames=var_skill_runs + arima_skill_runs,
    labels=['VAR'+str(i) for i in range(len(var_skill_runs))] +
        ['ARIMA'+str(i) for i in range(len(arima_skill_runs))],
    types=['VAR' for i in range(len(var_skill_runs))] +
           ['ARIMA' for i in range(len(arima_skill_runs))],
    panel_indicators=[
        False for i in range(len(var_skill_runs + arima_skill_runs))
    ],
    title='VAR_ARIMA ensemble overall 2023 rerun skill level 09262023',
    batch_names = ['batch_ARIMA top grid search runs 2023','batch_VAR top grid search runs 2023 rerun'],
    hierarchy_lvl='skill',
    model_selection= 'weighted average',
    output_occ_codes=True,
    rerun_2023= True,
    do_results_analysis= True
)





# old runs using just the best model
# create_ensemble_results(
#     runnames=[
#         'predicted job posting shares 15_53_15_17_04_2023 VAR for presentation lvl category',
#         'predicted job posting shares 15_51_36_17_04_2023 ARIMA for presentation lvl category',
#     ],
#     labels=[
#         'VAR',
#         'ARIMA',
#     ],
#     panel_indicators=[
#         False,
#         False,
#     ],
#     title='VAR_ARIMA ensemble category',
#     hierarchy_lvl='category',
# )
#
# create_ensemble_results(
#     runnames=[
#         'predicted job posting shares 16_10_03_17_04_2023 VAR for presentation lvl subcategory',
#         'predicted job posting shares 15_52_07_17_04_2023 ARIMA for presentation lvl subcategory',
#     ],
#     labels=[
#         'VAR',
#         'ARIMA',
#     ],
#     panel_indicators=[
#         False,
#         False,
#     ],
#     title='VAR_ARIMA ensemble subcategory',
#     hierarchy_lvl='subcategory',
# )
#
# create_ensemble_results(
#     runnames=[
#         'predicted job posting shares 16_43_47_17_04_2023 VAR for presentation lvl skill',
#         'predicted job posting shares 16_48_14_17_04_2023 ARIMA for presentation lvl skill',
#     ],
#     labels=[
#         'VAR',
#         'ARIMA',
#     ],
#     panel_indicators=[
#         False,
#         False,
#     ],
#     title='VAR_ARIMA ensemble skill',
#     hierarchy_lvl='skill',
# )

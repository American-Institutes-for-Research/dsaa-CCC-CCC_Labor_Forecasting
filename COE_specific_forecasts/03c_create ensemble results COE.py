'''
03c_create ensemble results.py
Luke Patterson
04-15-2023

calls to create ensemble results between two or more different runs based on best RMSE
'''

from utils import create_ensemble_results
import pandas as pd
import os
basepath = 'C:/AnacondaProjects/CCC forecasting'
os.chdir(basepath)

# coe_names = ['Business','Construction','Culinary & Hospitality','Education & Child Development','Engineering & Computer Science',
#              'Health Science','Information Technology','Manufacturing','Transportation, Distribution, & Logistics']
coe_names = ['Construction','Business','Culinary & Hospitality','Education & Child Development','Engineering & Computer Science',
             'Health Science','Information Technology','Manufacturing','Transportation, Distribution, & Logistics']

var_runs = os.listdir('output/batch_COE VAR runs v2')
arima_runs = os.listdir('output/batch_COE ARIMA runs v2')
var_skill_runs = [i.replace('.csv','') for i in var_runs if 'skill' in i]
var_subcat_runs = [i.replace('.csv','') for i in var_runs if 'subcategory' in i]
var_cat_runs = [i.replace('.csv','') for i in var_runs if ' category' in i]
arima_skill_runs = [i.replace('.csv','') for i in arima_runs if 'skill' in i]
arima_subcat_runs = [i.replace('.csv','') for i in arima_runs if 'subcategory' in i]
arima_cat_runs = [i.replace('.csv','') for i in arima_runs if ' category' in i]

for coe in coe_names:
    print(coe)
    full_raw_df = pd.read_csv("data/COE/test monthly counts " + coe + " combined.csv", index_col=0)
    cat_raw_df = full_raw_df[[i for i in full_raw_df.columns if 'Skill cat:' in i]]
    scat_raw_df = full_raw_df[[i for i in full_raw_df.columns if 'Skill subcat:' in i]]
    skill_raw_df = full_raw_df[[i for i in full_raw_df.columns if 'Skill:' in i]]
    cat_act_df = pd.read_csv('data/test monthly counts season-adj ' + coe + ' category.csv', index_col=0)
    scat_act_df = pd.read_csv('data/test monthly counts season-adj ' + coe + ' subcategory.csv', index_col=0)
    skill_act_df = pd.read_csv('data/test monthly counts season-adj ' + coe + ' skill.csv', index_col=0)
    var_coe_cat_runs = [i for i in var_cat_runs if coe in i]
    arima_coe_cat_runs = [i for i in arima_cat_runs if coe in i]


    create_ensemble_results(
        runnames=var_coe_cat_runs + arima_coe_cat_runs,
        labels=['VAR' + str(i) for i in range(len(var_coe_cat_runs))] +
               ['ARIMA' + str(i) for i in range(len(arima_coe_cat_runs))],
        types=['VAR' for i in range(len(var_coe_cat_runs))] +
               ['ARIMA' for i in range(len(arima_coe_cat_runs))],
        panel_indicators=[
            False for i in range(len(var_coe_cat_runs + arima_coe_cat_runs))
        ],
        title='VAR_ARIMA '+ coe +' category level',
        hierarchy_lvl='category',
        model_selection= 'weighted average',
        output_occ_codes= True,
        do_results_analysis= True,
        use_coe_fcast_folder= True,
        log_folder= 'result_logs/batch_all COE logs/',
        act_df= cat_act_df,
        raw_df= cat_raw_df
    )

    var_coe_subcat_runs = [i for i in var_subcat_runs if coe in i]
    arima_coe_subcat_runs = [i for i in arima_subcat_runs if coe in i]
    create_ensemble_results(
        runnames=var_coe_subcat_runs + arima_coe_subcat_runs,
        labels=['VAR'+str(i) for i in range(len(var_coe_subcat_runs))] +
            ['ARIMA'+str(i) for i in range(len(arima_coe_subcat_runs))],
        types=['VAR' for i in range(len(var_coe_subcat_runs))] +
            ['ARIMA' for i in range(len(arima_coe_subcat_runs))],
        panel_indicators=[
            False for i in range(len(var_coe_subcat_runs + arima_coe_subcat_runs))
        ],
        title='VAR_ARIMA ensemble '+coe+' subcategory level',
        hierarchy_lvl='subcategory',
        model_selection='weighted average',
        output_occ_codes=True,
        do_results_analysis=True,
        use_coe_fcast_folder=True,
        log_folder= 'result_logs/batch_all COE logs/',
        act_df=scat_act_df,
        raw_df=scat_raw_df
    )

    var_coe_skill_runs = [i for i in var_skill_runs if coe in i]
    arima_coe_skill_runs = [i for i in arima_skill_runs if coe in i]
    create_ensemble_results(
        runnames=var_coe_skill_runs + arima_coe_skill_runs,
        labels=['VAR'+str(i) for i in range(len(var_coe_skill_runs))] +
            ['ARIMA'+str(i) for i in range(len(arima_coe_skill_runs))],
        types=['VAR' for i in range(len(var_coe_skill_runs))] +
               ['ARIMA' for i in range(len(arima_coe_skill_runs))],
        panel_indicators=[
            False for i in range(len(var_coe_skill_runs + arima_coe_skill_runs))
        ],
        title='VAR_ARIMA ensemble '+coe+' skill level',
        hierarchy_lvl='skill',
        model_selection= 'weighted average',
        output_occ_codes=True,
        do_results_analysis=True,
        use_coe_fcast_folder=True,
        log_folder= 'result_logs/batch_all COE logs/',
        act_df=skill_act_df,
        raw_df=skill_raw_df
    )





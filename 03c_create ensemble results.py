'''
03c_create ensemble results.py
Luke Patterson
04-15-2023

calls to create ensemble results between two or more different runs based on best RMSE
'''

from utils import create_ensemble_results




create_ensemble_results(
    runnames=[
        'predicted job posting shares 15_53_15_17_04_2023 VAR for presentation lvl category',
        'predicted job posting shares 15_51_36_17_04_2023 ARIMA for presentation lvl category',
    ],
    labels=[
        'VAR',
        'ARIMA',
    ],
    panel_indicators=[
        False,
        False,
    ],
    title='VAR_ARIMA ensemble category',
    hierarchy_lvl='category',
)

create_ensemble_results(
    runnames=[
        'predicted job posting shares 16_10_03_17_04_2023 VAR for presentation lvl subcategory',
        'predicted job posting shares 15_52_07_17_04_2023 ARIMA for presentation lvl subcategory',
    ],
    labels=[
        'VAR',
        'ARIMA',
    ],
    panel_indicators=[
        False,
        False,
    ],
    title='VAR_ARIMA ensemble subcategory',
    hierarchy_lvl='subcategory',
)

create_ensemble_results(
    runnames=[
        'predicted job posting shares 16_43_47_17_04_2023 VAR for presentation lvl skill',
        'predicted job posting shares 16_48_14_17_04_2023 ARIMA for presentation lvl skill',
    ],
    labels=[
        'VAR',
        'ARIMA',
    ],
    panel_indicators=[
        False,
        False,
    ],
    title='VAR_ARIMA ensemble skill',
    hierarchy_lvl='skill',
)

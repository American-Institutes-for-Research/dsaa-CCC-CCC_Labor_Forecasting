from utils import compare_results

compare_results(
    runnames=[
        'predicted job posting shares 13_15_13_02_2023 DLPM 6 month input lvl category',
        'predicted job posting shares 09_46_26_01_2023 VAR no differencing 6 month input lvl category'
    ],
    labels=[
        'DLPM',
        'VAR'
    ],
    panel_indicators=[
        True,
        False
    ],
    title = 'VAR_DLPM comparison',
    hierarchy_lvl= 'category'
)

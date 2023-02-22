from utils import compare_results

compare_results(
    runnames=[
        'predicted job posting shares 15_29_15_02_2023 DLPM 6 month input lvl category',
        'predicted job posting shares 09_46_26_01_2023 VAR no differencing 6 month input lvl category',
        'predicted job posting shares 10_41_26_01_2023 Transformer no differencing 6 month input lvl category'
    ],
    labels=[
        'panel',
        'VAR',
        'transformer'
    ],
    panel_indicators=[
        True,
        False,
        False
    ],
    title = 'VAR_DLPM_ML comparison',
    hierarchy_lvl= 'category'
)

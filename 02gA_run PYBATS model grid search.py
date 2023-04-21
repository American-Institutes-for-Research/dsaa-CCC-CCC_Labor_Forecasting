from model_pybats_forecast_loop import run_pybats_loop
from utils import grid_search
grid_search(
    params_grid= {
        'k':[1,3,6,9,12],
        'rho':[.01,.1,.3,.5,.7,.9,.99],
        'deltrend':[.01,.1,.3,.5,.7,.9,.99],
        'delregn':[.01,.1,.3,.5,.7,.9,.99],
    },
    default_params= {
        'hierarchy_lvl':'category',
        'ccc_taught_only':False,
        'test_split':.25
    },
    loop_func= run_pybats_loop,
    batch_name = 'pybats grid search'
)


from skforecast_forecast_loop import run_skforecast_loop
from sklearn.ensemble import GradientBoostingRegressor

run_skforecast_loop(hierarchy_lvl='category', run_name='MLP test', param_grid_search=True)
#r = GradientBoostingRegressor()
#run_skforecast_loop(hierarchy_lvl='category', run_name='MLP test', param_grid_search=True, regressor = r)

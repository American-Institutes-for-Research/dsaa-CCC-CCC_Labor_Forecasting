from model_skforecast_forecast_loop import run_skforecast_loop
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

#run_skforecast_loop(hierarchy_lvl='category', run_name='MLP test', param_grid_search=True)
#r = GradientBoostingRegressor()
r = Lasso()
run_skforecast_loop(hierarchy_lvl='category', run_name='Lasso test', regressor = r)

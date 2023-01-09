# compare rankings between skill demand for VAR and ML rankings.
import pandas as pd

ml_df = pd.read_csv('output/predicted changes  17_19_12_12_2022 3 month input length.csv')
var_df = pd.read_csv("output/predicted changes  18_31_22_12_2022 no filtered skills VAR.csv")
var_df = var_df.loc[var_df['Monthly average obs'] > 50]

ml_df = ml_df.reset_index().rename({'index':'ML rank'}, axis = 1)
var_df = var_df.reset_index(drop=True).reset_index().rename({'index':'VAR rank'}, axis = 1)
ml_df = ml_df.set_index('Unnamed: 0', drop=True)
var_df = var_df.set_index('Unnamed: 0', drop=True)

rank_compare = ml_df[['ML rank']].merge(var_df[['VAR rank']], left_index=True, right_index=True)

print(rank_compare.corr(method='spearman'))
pass
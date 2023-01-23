# examine volatility of skill demand and correct for it

import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
# base counts
df = pd.read_csv('data/test monthly counts season-adj.csv', index_col=0)

# make sure that the positions in the dataframe corresponding to the first several months of covid are present
assert(df.index[19] == '2020-03-01')

# monthly change in counts
diff_df = df.diff().rolling(6).mean().dropna()

max_df = diff_df.idxmax()

# look at skills where changes were most volatile around the pandemic
covid_skills = max_df.loc[(max_df.str[:4] == '2020') & (max_df.str[5:7].astype('int') > 3)].index
covid_df = df.loc[:, covid_skills]
covid_diff_df = covid_df.diff().abs()

# average monthly observations
samp_df = pd.read_csv('working/average monthly observations by counts.csv', index_col = 0)
suff_skills = samp_df.loc[samp_df.iloc[:,0]>50]

# scale those skills with sufficient sample size to
#covid_diff_df = covid_diff_df.loc[:,[i for i in covid_diff_df.columns if i in suff_skills.index]]
min_max_scaler = preprocessing.MinMaxScaler()
norm_df = min_max_scaler.fit_transform(covid_diff_df)
norm_df = pd.DataFrame(norm_df, index= covid_diff_df.index, columns=covid_diff_df.columns)

# identify those with abnormal changes within the 9 months of 2020 in the pandemic
norm_df2 = norm_df.iloc[19:29,:].mean().sort_values(ascending=False)
norm_df2 = pd.DataFrame(norm_df2).merge(samp_df, left_index=True, right_index = True, how = 'left')
norm_df2.columns = ['covid volatility index', 'average monthly observations']
norm_df2.to_csv("working/skill covid volatility index.csv")
# norm_df2.hist()
# plt.show()

# dampen changes over time by the volatility in COVID
norm_df2['inverse index'] = 1 - norm_df2['covid volatility index']
diff_df2 = df.diff()

# multiply differences in 2020 covid months by the volatility index of the skill
mod_df = df.copy()
covid_months = diff_df2.index[19:29]
for s in norm_df2.index:
    diff_df2.loc[covid_months, s] = diff_df2.loc[covid_months,s] * norm_df2.loc[s, 'inverse index']
    mod_df.loc[covid_months, s] = mod_df.loc[mod_df.index[18], s] + diff_df2.loc[covid_months, s].cumsum()

pass
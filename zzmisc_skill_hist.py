import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/test monthly counts.csv", index_col= 0)
df = df.drop('Postings count', axis = 1)
assert all(["Skill" in c for c in df.columns])
df = df.iloc[7:55,:]

means = df.mean()
means = pd.Series([min(i, 110) for i in means], index = means.index)


means.hist()
plt.savefig('output/monthly mean skills count histogram.png')
import pandas as pd

df = pd.read_csv('data/postings/16980data_0_0_1.csv.gz')

df.to_csv('data/test row.csv')
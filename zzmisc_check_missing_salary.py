import pandas as pd
import os
import datetime

# import and append all data files

# tot_df = pd.read_csv('data/test monthly counts.csv')

filenum = len(os.listdir('data/us_postings'))
# n = 10
my_list = os.listdir('data/us_postings')
# tot_df = pd.read_csv('data/test monthly counts checkpoint.csv', index_col=0)
miss_counts = pd.Series([0,0],index=[True,False])
for i, f in enumerate(my_list):
    if i > -1 and '.csv.gz' in f:
        print('chunk', i, 'of', len(my_list), '-', f)
        df = pd.read_csv('data/us_postings/' + f)
        miss_counts = miss_counts + df.SALARY.isna().value_counts()

miss_counts.to_csv('working/Missing salary.csv')

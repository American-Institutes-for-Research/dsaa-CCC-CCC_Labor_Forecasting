# extract data on skills from us_postings_skills files
import pandas as pd
import os

full_df = pd.DataFrame()
for f in os.listdir('data/us_postings_skills'):
    if '.csv.gz' in f:
        df = pd.read_csv('data/us_postings_skills/'+f)
        df = df.drop('ID',axis=1)
        df = df.drop_duplicates()
        full_df = full_df.append(df)
        full_df = full_df.drop_duplicates()

full_df.to_csv('data/skill attributes data.csv')


import pandas as pd
import os
my_list = os.listdir('data/us_postings')
county_names = ['Cook, IL', 'DuPage, IL', 'Lake, IL', 'Will, IL', 'Kane, IL',
       'Lake, IN', 'McHenry, IL', 'Kenosha, WI', 'Porter, IN', 'DeKalb, IL',
       'Kendall, IL', 'Grundy, IL', 'Jasper, IN', 'Newton, IN']

for i, f in enumerate(my_list[0:2]):
    if '.csv.gz' in f:
        df = pd.read_csv('data/us_postings/' + f)
        df.COUNTY_NAME.value_counts().to_csv('working/sample county obs.csv')

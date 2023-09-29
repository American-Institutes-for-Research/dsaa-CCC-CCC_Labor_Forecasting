import pandas as pd
import os
basepath = 'C:/AnacondaProjects/CCC forecasting'
os.chdir(basepath)

coe_names = ['Business','Construction','Culinary & Hospitality','Education & Child Development','Engineering & Computer Science',
             'Health Science','Information Technology','Manufacturing','Transportation, Distribution, & Logistics']

for coe in coe_names:
    print(coe)
    df1 = pd.read_csv('data/COE/test monthly counts '+ coe + ' categories.csv', index_col = 0)
    df2 = pd.read_csv('data/COE/test monthly counts '+ coe + ' categories 2022-2023.csv', index_col = 0)
    df = df1 + df2
    combined_index = pd.Index(pd.Series(list(set(list(df1.index) + list(df2.index)))).sort_values().values)
    not_in_df1 = [i for i in combined_index if i not in df1.index]
    not_in_df2 = [i for i in combined_index if i not in df2.index]
    df1_supp = pd.DataFrame(index= not_in_df1, columns = df1.columns)
    df1_supp = df1_supp.fillna(0)
    df2_supp = pd.DataFrame(index=not_in_df2, columns=df2.columns)
    df2_supp = df2_supp.fillna(0)
    df1 = pd.concat([df1, df1_supp])
    df2 = pd.concat([df2, df2_supp])
    common_cols = [i for i in df1.columns if i in df2.columns]
    df1_only_cols = [i for i in df1.columns if i not in df2.columns]
    df2_only_cols = [i for i in df2.columns if i not in df1.columns]
    df = df1[common_cols] + df2[common_cols]
    df = pd.concat([df, df1[df1_only_cols], df2[df2_only_cols]], axis =1)
    df.to_csv('data/COE/test monthly counts ' + coe + ' combined.csv')

import pandas as pd
from matplotlib import pyplot as plt
from datetime import date

existing_index = pd.to_datetime(['2018-08-01', '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01',
                                '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01',
                                '2019-06-01', '2019-07-01', '2019-08-01', '2019-09-01', '2019-10-01',
                                '2019-11-01', '2019-12-01', '2020-01-01', '2020-02-01', '2020-03-01',
                                '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
                                '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01', '2021-01-01',
                                '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01',
                                '2021-07-01', '2021-08-01', '2021-09-01', '2021-10-01', '2021-11-01',
                                '2021-12-01', '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01',
                                '2022-05-01', '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01',
                                '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01', '2023-02-01',
                                '2023-03-01', '2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01'],
                               format='%Y-%m-%d')

# Create a new index with the desired end date
new_index = pd.date_range(start=existing_index[0], end='2025-07-01', freq='MS')

# Convert the new index to a list of date strings
new_index_list = new_index.strftime('%Y-%m-%d')

# If you want to convert the list to a Pandas Index with the name 'date':
new_index = pd.Index(new_index_list, name='date')

df = pd.read_csv('data/test monthly counts season-adj Health Science skill.csv',index_col=0)
df = df.reindex(index=pd.Index(new_index_list, name='date'))
df.index = [i.replace('-01','') for i in df.index]

df2 = pd.read_csv('data/test monthly counts season-adj skill.csv',index_col=0)
df2 = df2.reindex(index=pd.Index(new_index_list, name='date'))
df2.index = [i.replace('-01','') for i in df2.index]

df['Skill: Nursing'].plot()
plt.title('Nursing, Health Science COE occs only')
plt.xticks(rotation=45)
plt.xlabel('month')
plt.ylabel('job share')
plt.scatter(60+24,.403511419,label='predicted', color='red', marker='x')
plt.savefig('output/exhibits/nursing COE.png')
plt.clf()

df['Skill: Home Health Care'].plot()
plt.title('Home Health Care, Health Science COE occs only')
plt.scatter(60+24,.068428153,label='predicted', color='red', marker='x')
plt.xticks(rotation=45)
plt.xlabel('month')
plt.ylabel('job share')
plt.savefig('output/exhibits/home health care COE.png')
plt.clf()

df2['Skill: Nursing'].plot()
plt.title('Nursing, All occs')
plt.xticks(rotation=45)
plt.xlabel('month')
plt.ylabel('job share')
plt.scatter(60+24,.090101395,label='predicted', color='red', marker='x')
plt.savefig('output/exhibits/nursing overall.png')
plt.clf()

df2['Skill: Home Health Care'].plot()
plt.title('Home Health Care, All occs')
plt.scatter(60+24,.011512497,label='predicted', color='red', marker='x')
plt.xticks(rotation=45)
plt.xlabel('month')
plt.ylabel('job share')
plt.savefig('output/exhibits/home health care overall.png')
plt.clf()
pass

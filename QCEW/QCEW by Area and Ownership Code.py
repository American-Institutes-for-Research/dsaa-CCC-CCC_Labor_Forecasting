#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas


# In[3]:


# https://www.bls.gov/cew/downloadable-data-files.htm - "CSVs by Area" table heading
# All counties in U.S. already downloaded (2015-2022)
c21 = pandas.read_csv('C:/WDEMP/QCEW/by area/2021.q1-q4.by_area/2021.q1-q4 17031 Cook County, Illinois.csv')
c20 = pandas.read_csv('C:/WDEMP/QCEW/by area/2020.q1-q4.by_area/2020.q1-q4 17031 Cook County, Illinois.csv')
c19 = pandas.read_csv('C:/WDEMP/QCEW/by area/2019.q1-q4.by_area/2019.q1-q4 17031 Cook County, Illinois.csv')
c18 = pandas.read_csv('C:/WDEMP/QCEW/by area/2018.q1-q4.by_area/2018.q1-q4 17031 Cook County, Illinois.csv')
c17 = pandas.read_csv('C:/WDEMP/QCEW/by area/2017.q1-q4.by_area/2017.q1-q4 17031 Cook County, Illinois.csv')
c16 = pandas.read_csv('C:/WDEMP/QCEW/by area/2016.q1-q4.by_area/2016.q1-q4 17031 Cook County, Illinois.csv')
c15 = pandas.read_csv('C:/WDEMP/QCEW/by area/2015.q1-q4.by_area/2015.q1-q4 17031 Cook County, Illinois.csv')


# In[182]:


def naics_ts(df, naics, columns=['total_qtrly_wages']):
    """Creates time-series dataframe of QCEW data by user-supplied NAICS code
    df: QCEW by area file (imported above)
    naics: str for NAICS code, 2-6 digit. Will return blank df if NAICS code is not found.
    columns: list type, either 'total_qtrly_wages', 'qtrly_estabs_count', or 'monthX_emplvl'"""
    
    year = df.year.astype(str).values[0]
    df = df[df['industry_code']==naics]
    df = df.sort_values(by=['year','qtr'])
    df['yr.qtr'] = df.year.astype(str).str[2:4] + '.' + df.qtr.astype(int).astype(str)
    df.index = df['yr.qtr']
    out = df[columns]
    if len(df) == 0:
        if columns == ['total_qtrly_wages']:
            print('Missing', naics, 'in', year)
        yr = year[2:4]
        shell = pandas.DataFrame(columns=columns, index=[yr+'.1',yr+'.2',yr+'.3',yr+'.4'])
        shell.index.name = 'yr.qtr'
        return(shell)
    else:
        return(out)


# In[84]:


def stretch_month(df):
    """Helper function for monthX_emplvl columns"""
    year = df.index[0][0:2]
    df = df[['month1_emplvl','month2_emplvl','month3_emplvl']]
    vals = []
    for i in [df[j][k] for k in range(0,len(df)) for j in df.columns]:
        vals.append(i)
    new = {}
    vals_count = 0
    for q in range(1,5):
        for m in range(1, 4):
            new[year+'.'+str(q)+'.'+str(m)] = vals[vals_count]
            vals_count += 1
    out = pandas.DataFrame.from_dict(new, orient='index', columns=['employment'])
    out.index.rename('yr.qtr.month', inplace=True)
    return(out)


# In[85]:


def naics_ts_wrap(dfs, naics, columns):
    """Wrapper"""
    all_years = []
    for df in dfs:
        year = naics_ts(df, naics, columns)
        if len(columns) > 1:
            year = stretch_month(year)
        all_years.append(year)
    out = pandas.concat(all_years)
    if len(columns) > 1:
        out = out.sort_values(by='yr.qtr.month', ascending=True)
    else:
        out = out.sort_values(by='yr.qtr', ascending=True)
    return(out)


# In[86]:


def qcew_by_naics(dfs, naics):
    """Loops over all dfs (identified by year)"""
    emp_vars = ['month1_emplvl','month2_emplvl','month3_emplvl']
    wage_var, estabs_var  = ['total_qtrly_wages'], ['qtrly_estabs_count']
    employment = naics_ts_wrap(dfs=dfs, naics=naics, columns=emp_vars)
    wages = naics_ts_wrap(dfs=dfs, naics=naics, columns=wage_var)
    estabs = naics_ts_wrap(dfs=dfs, naics=naics, columns=estabs_var)
    qtrly = wages.merge(estabs, how='left', left_index=True, right_index=True)
    employment['yr.qtr'] = employment.index.str[0:4]
    df = employment.merge(qtrly, how='left', left_on='yr.qtr', right_index=True)
    df = df.rename(columns={'employment':'emp_month','total_qtrly_wages':'wages_qtr','qtrly_estabs_count':'estabs_qtr'})
    df['naics'] = naics
    df = df[['yr.qtr','naics','emp_month','wages_qtr','estabs_qtr']]
    return(df)


# In[87]:


def all_naics_by_area(dfs):
    """Loop over all available NAICS codes in any of the year-level QCEW by area files"""
    all_naics = list(dfs[0].industry_code.unique())
    for df in dfs:
        naics = list(df.industry_code.unique())
        addtl_naics = [x for x in naics if x not in all_naics]
        if len(addtl_naics) > 0:
            all_naics.extend(addtl_naics)
    all_naics.sort()
    store = []
    for code in all_naics:
        qn = qcew_by_naics(dfs, code)
        store.append(qn)
    out = pandas.concat(store)
    return(out)


# In[88]:


def filter_ownership(dfs, ownership='private'):
    own_codes = {'private':5, 'local':3, 'state':2, 'federal':1, 'all':0}
    out_dfs = []
    for df in dfs:
        out = df[df['own_code']==own_codes[ownership]]
        out_dfs.append(out)
    return(out_dfs)


# In[118]:


cook_dfs = [c21,c20,c19,c18,c17,c16,c15]
private = filter_ownership(cook_dfs)
local = filter_ownership(cook_dfs, 'local')
fed = filter_ownership(cook_dfs, 'federal')
state = filter_ownership(cook_dfs, 'state')


# In[176]:


cook_private = all_naics_by_area(private)


# In[177]:


cook_local = all_naics_by_area(local)


# In[178]:


cook_state = all_naics_by_area(state)


# In[174]:


cook_fed = all_naics_by_area(fed)


# In[179]:


# Concat, export
cook_private['ownership'] = 'private'
cook_state['ownership'] = 'state'
cook_local['ownership'] = 'local'
cook_fed['ownership'] = 'federal'
cook = pandas.concat([cook_private, cook_state, cook_local, cook_fed])
cook = cook.reset_index()
cook = cook.sort_values(by=['naics', 'ownership', 'yr.qtr.month'])
cook = cook.reset_index(drop=True)
cook.to_excel('C:/WDEMP/QCEW/output/QCEW_Cook_2015_2021.xlsx')


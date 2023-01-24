#!/usr/bin/env python
# coding: utf-8
import pandas, numpy

def process_oews(path, stats=True, translate=False):
    df = pandas.read_excel(path)
    df.columns = df.columns.str.lower()
    df.naics, df.occ_code = df.naics.astype(str), df.occ_code.astype(str)
    # some NAICS codes have letters, specifying a grouping of lower-level codes unique to OEWS data
    if translate:
        df = translate_naics_letters(df)
    df['pair'] = df.naics + '.' + df.occ_code
    if stats:
        print('Rows:', len(df))
        print('NAICS:', len(df.naics.unique()))
        print('O*Net:', len(df.occ_code.unique()))
        print('Pairings:', len(df.pair.unique()))
    return(df)

def translate_naics_letters(df):
   df['naics_has_letter'] = df['naics'].str.isdigit().replace({True:0, False:1})
   new_naics = df[df['naics_has_letter']==1].naics.unique().tolist()
   codes_titles = {}
   for code in new_naics:
       naics_title = df[df['naics']==code].naics_title.tolist()[0]
       nt = naics_title.split('(')[-1]
       for s in ['only',')',' ']:
           nt = nt.replace(s, '')
       nt = nt.replace('and',',')
       nt = nt.replace(',,',',')
       codes_titles[code] = nt
   df['naicsA'] = df['naics'].map(codes_titles)
   return df


# #### Nationwide Distributional Stats on Wages and Earnings for many pairings of O*Net codes and NAICS

folder = 'C:/WDEMP/OES/by NAICS/oesm21in4/'
print('Selected 6-digit NAICS')
dd = process_oews(folder+'nat5d_6d_M2021_dl.xlsx')
print('')
print('All 4-digit NAICS')
da = process_oews(folder+'nat4d_M2021_dl.xlsx', translate=True)
print('')
print('All 3-digit NAICS')
ds = process_oews(folder+'nat3d_M2021_dl.xlsx')

# Combining nationwide estimates at all NAICS levels (3,4,6)
df = pandas.concat([dd, da, ds])
df = df.reset_index(drop=True)
df['geography'] = 'US'
keep_cols = ['geography','naics','naics_title','naicsA','i_group','own_code','occ_code','occ_title','o_group','pair','tot_emp','emp_prse','pct_total','pct_rpt','h_mean','h_pct10','h_pct25','h_median','h_pct75','h_pct90','a_pct10', 'a_pct25','a_median','a_pct75','a_pct90']
rename = {'h_mean':'hourly_mean','h_pct10':'hourly_pct10','h_pct25':'hourly_pct25','h_median':'hourly_median','h_pct75':'hourly_pct75','h_pct90':'hourly_pct90','a_pct10':'yearly_pct10','a_pct25':'yearly_pct25','a_median':'yearly_median','a_pct75':'yearly_pct75','a_pct90':'yearly_pct90'}
df = df[keep_cols]
df = df.rename(columns=rename)


# #### Illinois wages/earnings for 797 O*Net codes, but no industry pairings

il_all = process_oews('C:/WDEMP/OES/by NAICS/oesm21st/state_M2021_dl.xlsx')
il = il_all[il_all['area_title']=='Illinois']
#keep_cols.remove('naicsA')
il['geography'] = 'Illinois'
il = il[keep_cols]
il = il.rename(columns=rename)

all = pandas.concat([df, il])
all.to_excel('C:/WDEMP/OES/output/wages_earnings_for_onet_naics_pairings.xlsx')

print('Rows:', len(all))
print('NAICS:', len(all.naics.unique()))
print('O*Net:', len(all.occ_code.unique()))
print('Pairings:', len(all.pair.unique()))

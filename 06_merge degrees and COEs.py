'''
06_merge degrees and COEs.py
Luke Patterson, 7-26-2023
Purpose: Conduct fuzzy matching on degree programs and COEs
'''
import pandas as pd
coe_df = pd.read_excel('emsi_skills_api/Academic Programs and Plans by COE - SU23.xlsx', skiprows=3)

# drop some COE degrees that don't appear in the degree data
coe_df = coe_df.drop(195)
coe_df = coe_df.loc[~coe_df['Plan Descr'].str.contains('Pre-')]
coe_df = coe_df.loc[~coe_df['Plan Descr'].str.contains('IOS and MACOS Development')]
coe_df = coe_df.loc[~coe_df['Plan Descr'].str.contains('Comp Aid Design Eng Tech AAS')]

# forward fill COE
coe_df['COE'] = coe_df['COE'].ffill()

degree_df = pd.read_excel('emsi_skills_api/CCC_degrees.xlsx', index_col=0)

def get_initials(s):
    s = s.replace(' in ',' ')
    words = s.split(' ')
    initials = ''.join([i[0] for i in words])
    return initials

degree_df['degree_abbr'] = degree_df.degree.apply(get_initials)
coe_df['plan_title'] = coe_df['Plan Descr'].apply(lambda x: x.split('-')[0])
coe_df['plan_initials'] = coe_df['Plan Descr'].apply(lambda x: x.split('-')[1])

# see what COEs get merged off of plan title and initials
degree_df = degree_df.merge(coe_df[['plan_title','plan_initials','COE']], how='left', left_on=['name','degree_abbr'],
                            right_on = ['plan_title','plan_initials'])



# output for manual matching
degree_df = degree_df.drop(['plan_title','plan_initials'], axis=1)
degree_df.to_excel('emsi_skills_api/CCC_degrees partial COE match.xlsx')

# keep just the degrees that have not been matched already
coe_df = coe_df.merge(degree_df[['name','degree_abbr']],how='left',right_on=['name','degree_abbr'],
                            left_on = ['plan_title','plan_initials'])
coe_df = coe_df.loc[coe_df.degree_abbr.isna()]

coe_df.to_excel('emsi_skills_api/Remaining COEs to match.xlsx')

# next is a manual review of the two output files for additional matches
pass

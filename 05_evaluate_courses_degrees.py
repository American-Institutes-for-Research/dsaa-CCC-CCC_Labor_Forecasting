# evaluate course/degree skill demand/growth based on forecasted skill demand

import pandas as pd
import numpy as np
from ast import literal_eval

# load data sets
course_df = pd.read_excel('emsi_skills_api/CCC_courses.xlsx')
course_df = course_df.drop('Unnamed: 0', axis=1)
degree_df = pd.read_excel('emsi_skills_api/CCC_degrees.xlsx')
fcast_df = pd.read_csv("output/predicted changes_univariate.csv")
skill_df = pd.read_csv('data/skill attributes data.csv')

# add skill ID to the forecast dataframe
fcast_df = fcast_df.rename({'Unnamed: 0':'Skill'}, axis=1)
fcast_df['Skill'] = fcast_df.Skill.str[1:]
fcast_df = fcast_df.merge(skill_df[['SKILL_ID','SKILL_NAME']], how='left', left_on ='Skill', right_on='SKILL_NAME')

fcast_df = fcast_df.set_index('SKILL_ID')

#fcast_df.to_csv('working/predictions with skill id.csv')

# loop through all courses
course_df['demand_share']= np.nan
course_df['demand_growth']= np.nan
course_df['avg_demand_share']= np.nan
course_df['avg_demand_growth']= np.nan
for i, skills in enumerate(course_df.skills):
    share_result = 0
    growth_result = 0
    # loop through all skills and record changes in share and growth across all related skills
    skills = literal_eval(skills)
    count = 0
    for s in skills:
        if s in fcast_df.index.values:
            count += 1
            fcast_share = fcast_df.loc[s, 'July 2024 predicted']
            fcast_growth = fcast_df.loc[s, 'Percent change']
            share_result += fcast_share
            growth_result += fcast_share * fcast_growth
            pass
    # if results are 0, means no skills have any forecasts, so we will leave as missing.
    if share_result != 0:
        course_df.loc[i, 'demand_share'] = share_result
        course_df.loc[i, 'avg_demand_share'] = share_result/count
    if growth_result != 0:
        course_df.loc[i, 'demand_growth'] = growth_result
        course_df.loc[i, 'avg_demand_growth'] = growth_result/count
course_df = course_df.set_index('course_id')
course_df.to_csv('working/course demand evaluations.csv')

# aggregate metrics up to degree level
degree_df['demand_share']= np.nan
degree_df['demand_growth']= np.nan
degree_df['avg_demand_share']= np.nan
degree_df['avg_demand_growth']= np.nan
for i, courses in enumerate(degree_df.courses):
    courses = literal_eval(courses)
    courses = [i['id'] for i in courses]
    degree_demand_growth = 0
    degree_demand_share = 0
    count = 0
    for c in courses:
        count += 1
        demand_growth = course_df.loc[c, 'demand_growth']
        # avg_demand_growth = course_df.loc[c, 'avg_demand_growth']
        demand_share = course_df.loc[c, 'demand_share']
        # avg_demand_share = course_df.loc[c, 'avg_demand_share']
        degree_demand_share += demand_share
        degree_demand_growth += demand_growth

    if degree_demand_growth != 0:
        degree_df.loc[i, 'demand_share'] = degree_demand_share
        degree_df.loc[i, 'avg_demand_share'] = degree_demand_share/count
    if degree_demand_share != 0:
        degree_df.loc[i, 'demand_growth'] = degree_demand_growth
        degree_df.loc[i, 'avg_demand_growth'] = degree_demand_growth/count

degree_df.to_csv('working/degree demand evaluations.csv', index=False)
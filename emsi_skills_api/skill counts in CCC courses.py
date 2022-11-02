import pandas as pd

df = pd.read_excel('courses_skills.xlsx')

count_df = df['name'].value_counts().sort_values()

count_df.to_excel('course_skill_counts.xlsx')
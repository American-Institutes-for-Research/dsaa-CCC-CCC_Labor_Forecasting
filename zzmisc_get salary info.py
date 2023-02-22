# get mean salary info on each skill/category

import pandas as pd
import os

my_list = os.listdir('data/us_postings')
cat_df = pd.read_excel('emsi_skills_api/EMSI_skills_with_categories.xlsx')
cats = cat_df['category_clean'].dropna().unique()
scats = cat_df['subcategory_clean'].dropna().unique()
cats_dict = {key:list() for key in cats}
scats_dict = {key:list() for key in scats}

# create dictionaries of all created categories and the skills within the
first = True
for label, row in cat_df.iterrows():
    if not pd.isna(row.category_clean):
        cats_dict[row['category_clean']].append(row['name'])
    if not pd.isna(row.subcategory_clean):
        scats_dict[row['subcategory_clean']].append(row['name'])

results_df = pd.DataFrame()


# tot_df = pd.read_csv('data/test monthly counts checkpoint.csv', index_col=0)
first = True
for i, f in enumerate(my_list):
    if i > -1 and '.csv.gz' in f:
        print('chunk', i, 'of', len(my_list), '-', f)
        df = pd.read_csv('data/us_postings/' + f)
        df = df.loc[~df.SALARY.isna()]
        df['SKILLS_NAME'] = df['SKILLS_NAME'].apply(lambda x: eval(x))
        # make unique list of skills
        skills = list(set([item for sublist in df['SKILLS_NAME'].values for item in sublist]))
        print('Num of Skills:', len(skills))

        supp_df = pd.DataFrame(index = df.index, columns= ['Skill: ' + i for i in skills] +
                               ['Skill subcat: ' + i for i in scats] +
                               ['Skill cat: ' + i for i in cats]).fillna(0)
        df = pd.concat([df, supp_df], axis = 1)
        for n, s in enumerate(skills):
            has_skill = df.SKILLS_NAME.apply(lambda x: s in x)
            df.loc[has_skill, 'Skill: ' + s] = 1
        skill_cols = ['Skill: ' + s for s in skills]

        for n, c in enumerate(cats):
            # for each row, flag whether any skills in the list in SKILLS_NAME column match skills in skill categories
            has_cat = df.SKILLS_NAME.apply(lambda skills: any([i in skills for i in cats_dict[c]]))
            df.loc[has_cat, 'Skill cat: ' + c] = 1

        # repeat for subcategories
        for n, s in enumerate(scats):
            # for each row, flag whether any skills in the list in SKILLS_NAME column match skills in skill categories
            has_scat = df.SKILLS_NAME.apply(lambda skills: any([i in skills for i in scats_dict[s]]))
            df.loc[has_scat, 'Skill subcat: ' + s] = 1

        skill_cat_cols = ['Skill cat: ' + i for i in cats]
        skill_subcat_cols = ['Skill subcat: ' + i for i in scats]

        if first:
            rdf = pd.DataFrame(index = cats)
            rdf['Level'] = 'Category'
            rdf2 = pd.DataFrame(index= scats)
            rdf2['Level'] = 'Subcategory'
            rdf3 = pd.DataFrame(index = skills)
            rdf3['Level'] = 'Skill'
            results_df = pd.concat([rdf, rdf2, rdf3])
            results_df['Salary sum'] = 0
            results_df['Salary obs'] = 0
            first = False
        else:
            new_skills = [i for i in skills if i not in results_df.index]
            rdf3 = pd.DataFrame(index=new_skills)
            rdf3['Level'] = 'Skill'
            rdf3['Salary sum'] = 0
            rdf3['Salary obs'] = 0
            results_df = pd.concat([results_df, rdf3])

        # record means
        for s in skills:
            filt_df = df.loc[df['Skill: ' + s] == 1]
            results_df.loc[s, 'Salary sum'] = results_df.loc[s, 'Salary sum'] + filt_df['SALARY'].sum()
            results_df.loc[s, 'Salary obs'] = results_df.loc[s, 'Salary obs'] + filt_df['SALARY'].count()

        for s in scats:
            filt_df = df.loc[df['Skill subcat: ' + s] == 1]
            results_df.loc[s, 'Salary sum'] = results_df.loc[s, 'Salary sum'] + filt_df['SALARY'].sum()
            results_df.loc[s, 'Salary obs'] = results_df.loc[s, 'Salary obs'] + filt_df['SALARY'].count()

        for s in cats:
            filt_df = df.loc[df['Skill cat: ' + s] == 1]
            results_df.loc[s, 'Salary sum'] = results_df.loc[s, 'Salary sum'] + filt_df['SALARY'].sum()
            results_df.loc[s, 'Salary obs'] = results_df.loc[s, 'Salary obs'] + filt_df['SALARY'].count()

        results_df['Salary mean'] = results_df['Salary sum']/results_df['Salary obs']

        results_df.to_csv('data/mean salary for skills.csv')

from datetime import datetime, timedelta
import pandas as pd
from linearmodels import IVGMM
from dateutil.relativedelta import relativedelta

def run_DLPM_loop(result_log = None, pred_df = None, start_val= 0,
                         input_len_used = 12, targets_sample = None, min_month_avg = 50, min_tot_inc = 50, cand_features_num=20
                         , ccc_taught_only = True, run_name = ''):
    '''
    params:
        result_log - previous result log data frame
        pred_df - previous prediction results dataframe
        start_val - skill number to start at for interrupted runs
        input_len_used - how many months prior to train on
        targets_sample - length of subset of targets to train on; used for shortening runtimes of tests
        min_month_avg - minimum monthly average job postings for skill to be forecasted for
        min_tot_inc - minimum total increase between first and last observed month for skill to be forecasted for
        run_name - name to give run's log/results files.

    Function to test run DLPM model with various parameters, and understand runtime
    '''

    date_run = datetime.now().strftime('%H_%M_%d_%m_%Y')
    if result_log is None:
        result_log = pd.DataFrame()

    df = pd.read_csv('data/test monthly counts county panel season-adj.csv', index_col=0)

    #--------------------
    # Feature Selection
    #-------------------

    # look only for those skills with mean 50 postings, or whose postings count have increased by 50 from the first to last month monitored

    raw_df = pd.read_csv('data/test monthly counts.csv')
    raw_df = raw_df.rename({'Unnamed: 0': 'date'}, axis=1)
    raw_df = raw_df.fillna(method='ffill')
    # 7-55 filter is to remove months with 0 obs
    raw_df = raw_df.iloc[7:55, :].reset_index(drop=True)
    # normalize all columns based on job postings counts
    raw_df = raw_df.drop('date', axis=1)

    # identify those skills who have from first to last month by at least 50 postings
    demand_diff = raw_df.T.iloc[:, -1] - raw_df.T.iloc[:, 0]
    targets = raw_df.mean(numeric_only=True).loc[(raw_df.mean(numeric_only=True)>min_month_avg)|(demand_diff > min_tot_inc)].index

    date_idx = pd.DatetimeIndex(df.index.str.split("'").map(lambda x: x[3]))
    county_idx = pd.Index(df.index.str.split("'").map(lambda x: x[1]))
    df = df.set_index([county_idx, date_idx])

    # include on CCC-taught skills
    if ccc_taught_only:
        ccc_df = pd.read_excel('emsi_skills_api/course_skill_counts.xlsx')
        ccc_df.columns = ['skill', 'count']
        ccc_skills = ['Skill: ' + i for i in ccc_df['skill']]
        targets = set(ccc_skills).intersection(set(targets)).union({'Postings count'})
        targets = list(targets)
        targets.sort()

    targets = df.columns[1:]
    targets = targets[start_val:]
    if targets_sample is not None:
        targets = targets[:targets_sample]

    # add in COVID case count data
    # covid_df = pd.read_csv('data/NYT COVID us-counties clean.csv')
    # # add 0 rows for pre-covid years
    # for y in range(2018,2020):
    #     for m in range(1,13):
    #         covid_df = covid_df.append(pd.Series([y, m, 0], index= ['year','month','cases_change']), ignore_index = True)
    #
    # # reshape to match the features data set and merge with features data
    # covid_df = covid_df.sort_values(['year','month'])
    # covid_df = covid_df.iloc[7:,:]
    # covid_df.index = date_idx
    # covid_df = covid_df.drop(['year','month'],axis=1)
    # covid_df.columns = ['covid_cases']
    # targets = targets.union(['covid_cases'])

    # df = df.merge(covid_df, left_index = True, right_index = True)

    # --------------------------
    # Model Execution
    # --------------------------
    print('Number of targets:', len(targets))
    if pred_df is None:
        pred_df = pd.DataFrame()
    features_main = df.corr()
    for n, t in enumerate(targets):

        endog= df[t]

        # test using most correlated features
        features = features_main[t].abs().sort_values(ascending=False).dropna()
        features = features.drop(t).iloc[:cand_features_num + 1]

        exogs = df[list(features.index) + [t]]
        min_t = min(endog.index.get_level_values(1))  # starting period
        T = len(date_idx.drop_duplicates())  # total number of periods

        # Names for the original and differenced endog and exog vars
        ename = endog.name
        Dename = 'D' + ename
        xnames = exogs.columns.tolist()
        Dxnames = ['D' + x for x in xnames]

        # We'll store all of the data in a dataframe
        data = pd.DataFrame()
        data[ename] = endog
        data[Dename] = endog.groupby(level=0).diff()

        # Generate and store the lags of the differenced endog variable
        Lenames = []
        LDenames = []
        for k in range(1, input_len_used + 1):
            col = 'L%s%s' % (k, ename)
            colD = 'L%s%s' % (k, Dename)
            Lenames.append(col)
            LDenames.append(colD)
            data[col] = data[ename].shift(k)
            data[colD] = data[Dename].shift(k)

        # Store the original and the diffs of the exog variables
        for x in xnames:
            data[x] = exogs[x]
            data['D' + x] = exogs[x].groupby(level=0).diff()

        # Set up the instruments -- lags of the endog levels for different time periods
        instrnames = []
        for n, t in enumerate(date_idx.drop_duplicates()):
            for k in range(1, input_len_used + 1):
                col = 'ILVL_t%iL%i' % (n, k)
                instrnames.append(col)
                data[col] = endog.groupby(level=0).shift(k)
                data.loc[endog.index.get_level_values(1) != t, col] = 0

        dropped = data.dropna()
        dropped['CLUSTER_VAR'] = dropped.index.get_level_values(0)

        # make sure columns are not duplicated to ensure full rank
        zero_cols = dropped.sum().loc[dropped.sum()==0].index
        instrnames = [i for i in instrnames if i not in zero_cols]
        dupes = dropped[instrnames].sum().duplicated()
        instrnames = [i for i in instrnames if i not in dupes.loc[dupes].index and 'L1' in i]
        model = IVGMM(dropped[Dename], dropped[Dxnames], dropped[LDenames], dropped[instrnames].iloc[:,:12],
                           weight_type='clustered', clusters=dropped['CLUSTER_VAR'])
        model.fit()

        pass

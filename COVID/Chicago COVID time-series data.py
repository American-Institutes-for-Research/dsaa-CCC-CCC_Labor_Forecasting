#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas, numpy
# data source: https://data.cityofchicago.org/Health-Human-Services/COVID-19-Hospital-Capacity-Metrics/f3he-c6sv
df = pandas.read_csv('C:/WDEMP/COVID/COVID-19_Hospital_Capacity_Metrics.csv')
df.columns = df.columns.str.lower()
df['dt'] = pandas.to_datetime(df['date'])
del df['daily update pdf'], df['date']
df = df.sort_values(by='dt')
cols = {'date':'date', 'ventilators - total capacity':'vent_total', 'ventilators in use - total':'vent_used', 'ventilators in use - covid-19':'vent_used_all_covid', 'ventilators in use - covid-19 patients':'vent_used_covid', 'ventilators in use - covid-19 pui':'vent_used_covid_pui', 'ventilators in use - non-covid-19':'vent_used_noncovid', 'ventilators available - total':'vent_available', 'ventilators available - hospital':'vent_available_hospital', 'ventilators available - eamc cache':'vent_available_eamc', 'ventilator surge capacity':'vent_surge_capacity', 'icu beds - total capacity':'icu_total', 'icu beds - adult':'icu_adult_total', 'icu beds - pediatric':'icu_pediatric_total', 'icu beds in use - total':'icu_filled', 'icu beds in use - covid-19':'icu_filled_covid_total', 'icu beds in use - covid-19 patients':'icu_filled_covid', 'icu beds in use - covid-19 pui':'icu_filled_covid_pui', 'icu beds in use - non-covid-19':'icu_filled_non_covid', 'icu beds available - total':'icu_available', 'icu beds available - adult':'icu_available_adult', 'icu beds available - pediatric':'icu_available_pediatric', 'icu beds surge capacity - adult':'icu_surge_adult', 'icu beds surge capacity - pediatric':'icu_surge_pediatric', 'acute non-icu beds - total capacity':'acute_total', 'acute non-icu beds in use - total':'acute_used', 'acute non-icu beds in use - covid-19':'acute_used_all_covid', 'acute non-icu beds in use - covid-19 patients':'acute_used_covid', 'acute non-icu beds in use - covid-19 pui':'acute_used_covid_pui', 'acute non-icu beds in use- non-covid-19':'acute_used_non_covid', 'acute non-icu beds available - total':'acute_available', 'combined hospital beds in use - covid-19':'all_used_beds_covid', 'dt':'date'}
df = df.rename(columns=cols)
df['year_month'] = df.date.dt.year.astype(str) + '.' + df.date.dt.month.astype(str)
dfm = df.groupby(by='year_month').sum()
dfm = dfm.sort_values(by='year_month')
dfm.to_excel('C:/WDEMP/COVID/chicago_covid_monthly.xlsx')


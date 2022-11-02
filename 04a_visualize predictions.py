import pandas as pd
from matplotlib import pyplot as plt

def forecast_graph(pred, actual, label):
    pred.name = 'Predicted'
    actual.name = 'Actual'

    pred.plot.line()
    actual.plot.line()
    plt.legend()
    plt.title(label)
    plt.savefig('output/exhibits/'+label+'.png')
    plt.clf()

pred_df = pd.read_csv("output/predicted job posting shares 17_23_20_10_2022.csv", index_col=0)
act_df = pd.read_csv('data/test monthly counts.csv', index_col=0)
pred_df.index = pd.to_datetime(pred_df.index)
act_df.index = pd.to_datetime(act_df.index)

job_counts = act_df['Postings count'].copy()
act_df = act_df.divide(job_counts, axis=0)
act_df['Postings count'] = job_counts


for i in pred_df.columns[:10]:
    if i != 'Postings count':
        forecast_graph(pred_df[i], act_df[i], i.replace('Skill: ','')+' graph')

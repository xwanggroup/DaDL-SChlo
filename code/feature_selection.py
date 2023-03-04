import pdb

import numpy as np
import pandas as pd
import xgboost as xgb
import operator
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


if __name__ == '__main__':
    train = pd.read_csv("fusion_dataset.csv")

    y = train['label1']
    X = train.drop(['label1','label2','label3','label4','label5'], 1)

    model = XGBClassifier(n_estimator=100)
    model.fit(X, y)
    # print(len(model.feature_importances_))
    # print(sum(model.feature_importances_))
    importance = [[idx,score]  for idx,score in enumerate(model.feature_importances_)]
    importance=sorted(importance,key=lambda x:x[1],reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv("feat_importance.csv", index=False)

import pdb

import  pandas as pd
importance=pd.read_csv("feat_importance.csv")["feature"]
select_fea=list(importance)[0:2000]
select_fea=[str(i+1) for i in select_fea]
labels=['label1','label2','label3','label4','label5']
labels.extend(select_fea)
train = pd.read_csv("fusion_dataset.csv")
df = train[labels]
df.to_csv("feature_select.csv", index=False)

      

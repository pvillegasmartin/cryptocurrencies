import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans

#Sentiment data
df_sentiment = pd.read_csv(r'C:\Users\Pablo\Desktop\PMG\Strategies\Sentiment\data\augmento_btc.csv')
#df_sentiment['date'] = df_sentiment['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df_sentiment = df_sentiment.set_index('date')
#df_sentiment = df_sentiment[df_sentiment.index.year > 2018]
df_sentiment = df_sentiment.dropna()

# Checking that one platform values are not higher in general
# df_sentiment.mean().sort_values(ascending=False)

# Split data
X = df_sentiment.iloc[:,1:]
y = df_sentiment.iloc[:,0]

# feature selection
twitter_cols = []
reddit_cols = []
bitcointalk_cols = []
for col_name in X.columns:
    if 'twitter' in col_name:
        twitter_cols.append(col_name)
    elif 'reddit' in col_name:
        reddit_cols.append(col_name)
    elif 'bitcointalk' in col_name:
        bitcointalk_cols.append(col_name)

X_twitter = X[twitter_cols]
X_twitter.columns = X_twitter.columns.str.split('_').str[-1]
X_reddit = X[reddit_cols]
X_reddit.columns = X_reddit.columns.str.split('_').str[-1]
X_bitcointalk = X[bitcointalk_cols]
X_bitcointalk.columns = X_bitcointalk.columns.str.split('_').str[-1]

data = pd.concat((X_reddit, X_twitter))
data = pd.concat((data, X_bitcointalk))
data = data.groupby(data.index).mean()
del X_twitter, X_reddit, X_bitcointalk, df_sentiment

# feature selection
# Show all features shape
'''
n_cols = 5
n_rows = math.ceil(len(data.columns)/n_cols)
fig, axs = plt.subplots(n_rows,n_cols,figsize=(25,25))
fig.suptitle('Series')
for i in range(n_rows):
    for j in range(n_cols):
        if i*n_cols+j+1>len(data.columns):
            continue
        axs[i, j].plot(data.iloc[:,i*n_cols+j].values)
        axs[i, j].set_title(data.columns[i*n_cols+j])
plt.show()


fig, axs = plt.subplots(3,1,figsize=(25,25))
axs[0].plot(y)
axs[1].plot(data.bullish)
axs[2].plot(data.bearish)
plt.show()
'''
# pca = PCA(n_components=2)
# data_pca = pca.fit_transform(data)
print(23)
#
# model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10)
# model.fit(data)
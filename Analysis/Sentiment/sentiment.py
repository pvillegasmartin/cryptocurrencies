import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr

from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler

#Sentiment data
df_sentiment = pd.read_csv(r'C:\Users\Pablo\Desktop\PMG\Strategies\Sentiment\data\augmento_btc.csv')
df_sentiment['date'] = df_sentiment['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
df_sentiment = df_sentiment.set_index('date')
df_sentiment = df_sentiment[df_sentiment.index.year > 2018]
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

# EDA
fig, ax = plt.subplots()
ax.bar(data.index, data['positive'], label='Positive', color='green', linewidth=0.25)
ax.bar(data.index, data['negative'], label='Negative', color='red', alpha=0.5, linewidth=0.25)
ax.legend(loc='upper left')
ax2 = ax.twinx()
ax2.plot(y, label='BTC Price', linewidth=1)
ax2.legend(loc='upper right')
plt.show()

correlations_y = []
for feature in data.columns:
    try:
        corr = pearsonr(data[feature],y)[0]
        correlations_y.append([feature,corr])
    except:
        pass

# Feature selection
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

# SOM method - Self-organizing maps are a type of neural network that is trained using unsupervised learning to produce a low-dimensional representation of the input space of the training samples, called a map
def plot_som_series_averaged_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
    fig.suptitle('Clusters')
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5)
                axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")
    plt.show()

som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
som = MiniSom(som_x, som_y,len(train_scaled[0]), sigma=0.3, learning_rate = 0.1)
som.random_weights_init(train_scaled)
som.train(train_scaled, 50000)
win_map = som.win_map(train_scaled)
plot_som_series_averaged_center(som_x, som_y, win_map)


# K-means for time series
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10)
labels = model.fit_predict(mySeries)

plot_count = math.ceil(math.sqrt(3))
fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
fig.suptitle('Clusters')
row_i = 0
column_j = 0
# For each label there is, plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
        if (labels[i] == label):
            axs[row_i, column_j].plot(mySeries[i], c="gray", alpha=0.4)
            cluster.append(mySeries[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
    axs[row_i, column_j].set_title("Cluster " + str(row_i * plot_count + column_j))
    column_j += 1
    if column_j % plot_count == 0:
        row_i += 1
        column_j = 0
plt.show()

# K-means with previous dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(mySeries)
# Plot time series in 2 dimensions
# plt.figure(figsize=(25,10))
# plt.scatter(data_pca[:,0],data_pca[:,1], s=300)
# plt.show()

kmeans = KMeans(n_clusters=3,max_iter=10000)
labels = kmeans.fit_predict(data_pca)
# Plot time series in 2 dimensions with labels
# plt.figure(figsize=(25,10))
# plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, s=300)
# plt.show()
# Plot distribution labels
# cluster_c = [len(labels[labels==i]) for i in range(3)]
# cluster_n = ["cluster_"+str(i) for i in range(3)]
# plt.figure(figsize=(15,5))
# plt.title("Cluster Distribution for KMeans")
# plt.bar(cluster_n,cluster_c)
# plt.show()
# Which labels each serie
fancy_names_for_labels = [f"Cluster {label}" for label in labels]
series_classified = pd.DataFrame(zip(data.columns,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")
'''
# CORRELATION
row_train = int(len(data)*0.8)
train, test = data.iloc[:row_train,:], data.iloc[row_train:,:]
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

mySeries = np.transpose(train_scaled)
# Correlation
correlations = []
for feature in data.columns:
    for variable in data.columns:
        try:
            corr = pearsonr(data[feature],data[variable])[0]
            correlations.append([feature,variable,corr])
        except:
            pass
print(1)
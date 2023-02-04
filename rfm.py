import pandas as pd
import numpy as np

df_order = pd.read_csv('orders.txt', sep = '\t', encoding="ISO-8859-1", parse_dates = ['orderdate'])
df_customer = pd.read_csv('customer.txt', sep = '\t', encoding= "ISO-8859-1")

df_order = df_order[['orderid', 'customerid', 'orderdate', 'totalprice']]
df = df_order.merge(df_customer[['customerid', 'householdid']], left_on = 'customerid', right_on='customerid')
#print(df.groupby('householdid').agg({'orderid': lambda x: len(x)}))
# number of orders placed per household

df_1 = df.groupby('householdid')['orderdate'].max().reset_index()
# give an index, so col householdid is no longer an index
df_1.columns=[['householdid','max_date']]
# rename 2nd column as max_date

df_1['recency']= (df_1['max_date'].max() - df_1['max_date']).apply(lambda x: x.dt.days)
# series.dt.days changes date data type to integer data type, return # of days
#print(df_1)

# 这里不在武断分 5 groups for each R/F/M
# auto clustering with KMeans, input is recency
# EDA
import matplotlib.pyplot as plt
#plt.hist(df_1['recency'])
#plt.show()

from sklearn.cluster import KMeans
'''
# how to determine K?
sse = {}

for n in range(1,10):  # tried (1,2), nothing much happened,  now try (1,10), it gives more info and one elbow point
    kmeans = KMeans(n_clusters=n, random_state=0).fit(df_1['recency'].to_numpy())
            # fit expects ndarray
            # kmeans return labels, index of the cluster each sample belongs to
    df_1['clusters'] = kmeans.labels_  # which cluster the householdID belongs to?
    sse[n] = kmeans.inertia_
    # inertia_: return float, sum of squared distances of samples to closest cluster center

print(df_1)
print(sse.items()) # pair of key and value

# elbow method
plt.plot(sse.keys(), sse.values())
plt.show()
# saw 3 is the elbow point, then K in Kmeans for Recency is 3+1=4. We only need 4 clusters.
'''

# replace n with 4
# predict Recency with Recency test?
# train-test split
kmeans = KMeans(n_clusters=4, random_state=0).fit(df_1['recency'].to_numpy())
df_1['recency_cluster'] = kmeans.predict(df_1['recency'].to_numpy()) # return multi-level index
# predict(): Predict the closest cluster each sample in X belongs to.
# so far, clusters is nominal which has no rank, but we need ordinal clusters which we could rank clusters

# we get error from order_cluster function "ValueError: Grouper for 'recency_cluster' not 1-dimensional"
# because ['recency_cluster'] has multi-level index -->we need to flatten it, reducing to 1 dimension
df_1.columns= df_1.columns.get_level_values(0)

# create a function for ordering cluster numbers - make clusters ordinal
def order_cluster(cluster_field_name, target_field_name, df, ascending):

    # get means of recency for each cluster
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    # print(df_new)

    # sort rows by recency
    df_new = df_new.sort_values(by=target_field_name, ascending = ascending).reset_index(drop = True)
    df_new['index'] = df_new.index #add a new column
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    # df now has 2 more columns recency_cluster, and ordinal 'index'

    # remove column recency_cluster
    df_final = df_final.drop([cluster_field_name],axis = 1)

    # rename column "index" as "recency_cluster" since we want recency_cluster to be ordinal
    df_final = df_final.rename(columns = {"index": cluster_field_name})
    return df_final

df_orderd_by_recency = order_cluster('recency_cluster', 'recency', df_1, False)
print(df_orderd_by_recency)
# we've completed auto-clustering for Recency
# we need to the same for Frequency and Monetary Value















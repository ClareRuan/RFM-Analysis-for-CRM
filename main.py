import pandas as pd
import numpy as np

# import data
df = pd.read_csv('orders.txt', sep='\t', encoding="ISO-8859-1", parse_dates=['orderdate'])
# \t tells python the file has delimiter as TAB
# parse_dates: tell python this column is date type
# utf-8 mostly data North America, but other countries might be different, e.g. Europe
# encoding: tell python how to decode this file since we have utf-8 error

# EDA
#print(df)
#print(df.dtypes)

# modeling
# POC
# CLV = AOV * F * MG * (1/Churn Rate) - AC
margin = 0.05
ac = 1
# group clients based on order date
# for each cx, 找order date column 最大值和最小值, get the period, so we know how long this cx has stayed with the company
customers = df.groupby('customerid').agg({'orderdate': lambda x: (x.max() - x.min()).days,
                                          'totalprice': lambda x: x.sum(),
                                          'orderid': lambda x: len(x)})


retention = len(customers[customers['orderid'] > 1].index) / len(customers.index)

#  或者写成 shape[0], 注意是[], 不是()
# number of cx who placed order more than once /  total number of cx = retention
# len() and shape(0) is faster than count()

# days : change data type from date to numbers
customers= customers[customers['orderdate']>0]
# original order.txt only has one cx placed order more than once, customerid=0
avg_order = customers['totalprice'].sum()/customers['orderid'].sum()
# not len() because for each cx, we have aggregated sum on number of orders

freq = customers['orderid'].sum() / customers['orderdate'].sum()
# maybe there's different definitions on freq: Total Orders / Total Number of Unique Clients
# one definition: for each customer, how many orders placed per day during her time doing business with the company

print(customers)
print(f' avg_order: {avg_order} and freq: {freq}')

# get CLV
print( avg_order * freq * margin * (1-retention) - ac)

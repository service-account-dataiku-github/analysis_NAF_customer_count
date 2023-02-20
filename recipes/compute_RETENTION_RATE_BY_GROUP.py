# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import time

t0 = time.time()

print("loading...")

# Read recipe inputs
NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT = dataiku.Dataset("NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT")
NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT_df = NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT.get_dataframe()

NAFCUSTOMER_REVENUE_AGGREGATED = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED")
NAFCUSTOMER_REVENUE_AGGREGATED_df = NAFCUSTOMER_REVENUE_AGGREGATED.get_dataframe()

t1 = time.time()
print("load duration", (t1-t0)/60.0, "minutes")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT_df), "rows in NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT")
df_a = NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT_df.copy()

print(len(NAFCUSTOMER_REVENUE_AGGREGATED_df), "rows in NAFCUSTOMER_REVENUE_AGGREGATED")
df = NAFCUSTOMER_REVENUE_AGGREGATED_df.copy()
df = df[df.REVENUE_YEAR!=2023]
print(len(df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_revenue_by_customer_and_year = df.groupby(['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR']).REVENUE_AMOUNT_USD.sum().reset_index()
print(len(df_revenue_by_customer_and_year))

df_spend_by_customer_and_year = df.groupby(['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR']).GROSS_SPEND_AMOUNT.sum().reset_index()
print(len(df_spend_by_customer_and_year))

df_active_card_count_by_customer = df_a.groupby(['CUSTOMER_ID','CUSTOMER','YEAR_NUMBER']).ACTIVE_CARD_COUNT.sum().reset_index()
df_active_card_count_by_customer.columns = ['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR','ACTIVE_CARD_COUNT']
print(len(df_active_card_count_by_customer))

df_customer_fleet_size = df_active_card_count_by_customer.groupby(['CUSTOMER_ID','CUSTOMER']).ACTIVE_CARD_COUNT.max().reset_index()
df_customer_fleet_size.columns = ['CUSTOMER_ID', 'CUSTOMER', 'FLEET_SIZE']
print(len(df_customer_fleet_size))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df_revenue_by_customer_and_year))
df_j = pd.merge(df_revenue_by_customer_and_year,df_spend_by_customer_and_year,how='left',on=['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR'])
print(len(df_j))

df_j = pd.merge(df_j,df_active_card_count_by_customer,how='left',on=['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR'])
print(len(df_j))

df_j = pd.merge(df_j,df_customer_fleet_size, how='left',on=['CUSTOMER_ID','CUSTOMER'])
print(len(df_j))

df_j.loc[df_j.ACTIVE_CARD_COUNT.isnull(),'ACTIVE_CARD_COUNT'] = 0
df_j.loc[df_j.FLEET_SIZE.isnull(),'FLEET_SIZE'] = 0

df_j['FLEET_CATEGORY'] = 'NOT SET'
df_j.loc[df_j.FLEET_SIZE.between(0,5),'FLEET_CATEGORY'] = '(<=5 cards)'
df_j.loc[df_j.FLEET_SIZE.between(6,100),'FLEET_CATEGORY'] = '(between 6 and 100 cards)'
df_j.loc[df_j.FLEET_SIZE>50,'FLEET_CATEGORY'] = '(>100 cards)'
df_j.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_j.FLEET_CATEGORY.unique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_revenue_by_year = df_j.groupby(['REVENUE_YEAR','FLEET_CATEGORY']).REVENUE_AMOUNT_USD.sum().reset_index()
df_revenue_by_year.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_revenue_by_year.REVENUE_YEAR.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

fig, ax1 = plt.subplots(figsize=(9,4))

#['(<=5 cards)', '(between 6 and 100 cards)', '(>100 cards)']

dim1 = list(map(str,df_revenue_by_year[df_revenue_by_year.FLEET_CATEGORY=='(>100 cards)'].REVENUE_YEAR))
data1 = df_revenue_by_year[df_revenue_by_year.FLEET_CATEGORY=='(>100 cards)'].REVENUE_AMOUNT_USD

dim2 = list(map(str,df_revenue_by_year[df_revenue_by_year.FLEET_CATEGORY=='(between 6 and 100 cards)'].REVENUE_YEAR))
data2 = df_revenue_by_year[df_revenue_by_year.FLEET_CATEGORY=='(between 6 and 100 cards)'].REVENUE_AMOUNT_USD

dim3 = list(map(str,df_revenue_by_year[df_revenue_by_year.FLEET_CATEGORY=='(<=5 cards)'].REVENUE_YEAR))
data3 = df_revenue_by_year[df_revenue_by_year.FLEET_CATEGORY=='(<=5 cards)'].REVENUE_AMOUNT_USD

#ax1.bar(dim1,data1, color='#006BA2', width=0.75)
#ax1.bar(dim2,data2, bottom=data1, color='#758D99', width=0.75)
#ax1.bar(dim3,data3, bottom=data2+data1, color='#EBB434', width=0.75)

ax1.plot(dim1,data1, color='#006BA2')
ax1.plot(dim2,data2, color='#758D99')
ax1.plot(dim3,data3, color='#EBB434')

ax1.set_xlabel('YEAR', fontsize=14)
ax1.set_ylabel('REVENUE', fontsize=14)
#ax1.set_ylim(ymin=0, ymax=max_value*1.15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.6)
fig.autofmt_xdate()
plt.title('TURN THIS INTO A PROPER BAR')
plt.legend(['(>100 cards)','(between 6 and 100 cards)','(<=5 cards)'], bbox_to_anchor=(1.05,1.0), loc='upper left')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_spend_by_year = df_j.groupby(['REVENUE_YEAR','FLEET_CATEGORY']).GROSS_SPEND_AMOUNT.sum().reset_index()
df_spend_by_year.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

fig, ax1 = plt.subplots(figsize=(9,4))

dim1 = list(map(str,df_spend_by_year[df_spend_by_year.FLEET_CATEGORY=='(>20 cards)'].REVENUE_YEAR))
data1 = df_spend_by_year[df_spend_by_year.FLEET_CATEGORY=='(>20 cards)'].GROSS_SPEND_AMOUNT

dim2 = list(map(str,df_spend_by_year[df_spend_by_year.FLEET_CATEGORY=='(<=20 cards)'].REVENUE_YEAR))
data2 = df_spend_by_year[df_spend_by_year.FLEET_CATEGORY=='(<=20 cards)'].GROSS_SPEND_AMOUNT

ax1.bar(dim1,data1, color='#006BA2', width=0.75)
ax1.bar(dim2,data2, bottom=data1, color='#758D99', width=0.75)

ax1.set_xlabel('YEAR', fontsize=14)
ax1.set_ylabel('SPEND', fontsize=14)
#ax1.set_ylim(ymin=0, ymax=max_value*1.15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.6)
fig.autofmt_xdate()
plt.title('SPEND by YEAR')
plt.legend(['(>20 cards)','(<=20 cards)'], bbox_to_anchor=(1.05,1.0), loc='upper left')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_customer_by_year = df_j.groupby(['REVENUE_YEAR','FLEET_CATEGORY']).CUSTOMER_ID.nunique().reset_index()
df_customer_by_year.columns = ['REVENUE_YEAR', 'FLEET_CATEGORY', 'CUSTOMER_COUNT']
df_customer_by_year.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

fig, ax1 = plt.subplots(figsize=(9,4))

dim1 = list(map(str,df_customer_by_year[df_customer_by_year.FLEET_CATEGORY=='(>20 cards)'].REVENUE_YEAR))
data1 = df_customer_by_year[df_customer_by_year.FLEET_CATEGORY=='(>20 cards)'].CUSTOMER_COUNT

dim2 = list(map(str,df_customer_by_year[df_customer_by_year.FLEET_CATEGORY=='(<=20 cards)'].REVENUE_YEAR))
data2 = df_customer_by_year[df_customer_by_year.FLEET_CATEGORY=='(<=20 cards)'].CUSTOMER_COUNT

ax1.bar(dim1,data1, color='#006BA2', width=0.75)
ax1.bar(dim2,data2, bottom=data1, color='#758D99', width=0.75)

ax1.set_xlabel('YEAR', fontsize=14)
ax1.set_ylabel('CUSTOMER COUNT', fontsize=14)
#ax1.set_ylim(ymin=0, ymax=max_value*1.15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.6)
fig.autofmt_xdate()
plt.title('CUSTOMER COUNT by YEAR')
plt.legend(['(>20 cards)','(<=20 cards)'], bbox_to_anchor=(1.05,1.0), loc='upper left')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_active_cards_by_year = df_j.groupby(['REVENUE_YEAR','FLEET_CATEGORY']).ACTIVE_CARD_COUNT.sum().reset_index()
df_active_cards_by_year.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

fig, ax1 = plt.subplots(figsize=(9,4))

dim1 = list(map(str,df_active_cards_by_year[df_active_cards_by_year.FLEET_CATEGORY=='(>20 cards)'].REVENUE_YEAR))
data1 = df_active_cards_by_year[df_active_cards_by_year.FLEET_CATEGORY=='(>20 cards)'].ACTIVE_CARD_COUNT

dim2 = list(map(str,df_active_cards_by_year[df_active_cards_by_year.FLEET_CATEGORY=='(<=20 cards)'].REVENUE_YEAR))
data2 = df_active_cards_by_year[df_active_cards_by_year.FLEET_CATEGORY=='(<=20 cards)'].ACTIVE_CARD_COUNT

ax1.bar(dim1,data1, color='#006BA2', width=0.75)
ax1.bar(dim2,data2, bottom=data1, color='#758D99', width=0.75)

ax1.set_xlabel('YEAR', fontsize=14)
ax1.set_ylabel('ACTIVE CARDS', fontsize=14)
#ax1.set_ylim(ymin=0, ymax=max_value*1.15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.6)
fig.autofmt_xdate()
plt.title('ACTIVE CARD COUNT by YEAR')
plt.legend(['(>20 cards)','(<=20 cards)'], bbox_to_anchor=(1.05,1.0), loc='upper left')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

#RETENTION_RATE_BY_GROUP_df = ... # Compute a Pandas dataframe to write into RETENTION_RATE_BY_GROUP

# Write recipe outputs
#RETENTION_RATE_BY_GROUP = dataiku.Dataset("RETENTION_RATE_BY_GROUP")
#RETENTION_RATE_BY_GROUP.write_with_schema(RETENTION_RATE_BY_GROUP_df)
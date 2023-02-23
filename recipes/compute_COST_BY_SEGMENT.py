# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from datetime import date, datetime, timedelta
import time

import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,}'.format

t0 = time.time()

# Read recipe inputs
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019 = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019")
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019_df = NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019.get_dataframe()
print('loaded file 2019')

NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021 = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021")
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021_df = NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021.get_dataframe()
print('loaded file 2020')

NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020 = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020")
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020_df = NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020.get_dataframe()
print('loaded file 2021')

NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022 = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022")
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022_df = NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022.get_dataframe()
print('loaded file 2022')

NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT = dataiku.Dataset("NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT")
NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT_df = NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT.get_dataframe()
print('loaded annual card count')

HFM_cost_by_MRU = dataiku.Dataset("HFM_cost_by_MRU")
HFM_cost_by_MRU_df = HFM_cost_by_MRU.get_dataframe()
print('loaded HFM cost data')


t1 = time.time()
print("load duration", (t1-t0)/60.0, "minutes")

# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
row_count = 0
row_count += len(NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019_df)
row_count += len(NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020_df)
row_count += len(NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021_df)
row_count += len(NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022_df)
print(row_count)

df = pd.concat([NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019_df, NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020_df, 
               NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021_df, NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022_df])

print(len(df), "NAFCUSTOMER REVENUE AGGREGATED")

print(len(NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT_df), "NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT_df")
df_a = NAFCUSTOMER_ACCOUNT_ANNUAL_ACTIVE_CARD_COUNT_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['CUSTOMER_ID'] = df['CUSTOMER_ID'].astype('Int64')
df['SETUP_DATE_DT'] = pd.to_datetime(df['SETUP_DATE'])
df['SETUP_YEAR'] = df.SETUP_DATE_DT.dt.year

df['REVENUE_YEAR'] = df['REVENUE_YEAR'].astype('Int64')
df['REVENUE_DATE'] = df['REVENUE_MONTH'].astype(str) + '-' + df['REVENUE_YEAR'].astype(str)
df['REVENUE_DATE'] = pd.to_datetime(df['REVENUE_DATE'], format='%m-%Y').dt.strftime('%m-%Y')

df['MRU'] = df.BI_MRU
df.loc[df.MRU.isnull(),'MRU'] = 3100
df.loc[df.MRU=='TBD','MRU'] = 3100
df['MRU'] = df['MRU'].astype(float)
df['MRU'] = df['MRU'].astype('Int64')
df.MRU.value_counts()

print(len(df), "before filter")
df = df[df.REVENUE_YEAR!=2023]
print(len(df), "after filter, removing 2023")

# break down of revenue_year
df.REVENUE_YEAR.value_counts()

df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# plot out sum of revenue per year
# in order to do a quick reonciliation against

# aggregate revenue at annual grain
df_revenue_per_year = df.groupby(['REVENUE_YEAR']).REVENUE_AMOUNT_USD.sum().reset_index()
df_revenue_per_year.REVENUE_AMOUNT_USD = df_revenue_per_year.REVENUE_AMOUNT_USD
df_revenue_per_year.head()

max_revenue = df_revenue_per_year.REVENUE_AMOUNT_USD.max()

chart_revenue_year = []
for y in df_revenue_per_year.REVENUE_YEAR.tolist():
    chart_revenue_year.append(str(y))

fig, ax1 = plt.subplots(figsize=(8,3))
ax1.plot(chart_revenue_year,df_revenue_per_year['REVENUE_AMOUNT_USD'], marker='o')
ax1.set_xlabel('YEAR', fontsize=14)
ax1.set_ylabel('REVENUE (M)', fontsize=14)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
ax1.grid()
ax1.set_ylim(ymin=0, ymax=max_revenue*1.15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
fig.autofmt_xdate()
plt.title('Quick Revenue Reconciliation Table')
plt.show()

df_revenue_per_year.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_customer_MRU = df.groupby(['CUSTOMER_ID','CUSTOMER']).MRU.min().reset_index()
df_customer_program = df.groupby(['CUSTOMER_ID','CUSTOMER']).CUSTOMER_BUSINESS_PROGRAM_NAME.first().reset_index()

df_customer_revenue_by_year = df.groupby(['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR']).REVENUE_AMOUNT_USD.sum().reset_index()
print(len(df_customer_revenue_by_year))

df_active_card_count_by_customer = df_a.groupby(['CUSTOMER_ID','CUSTOMER','YEAR_NUMBER']).ACTIVE_CARD_COUNT.sum().reset_index()
df_active_card_count_by_customer.columns = ['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR','ACTIVE_CARD_COUNT']
df_active_card_count_by_customer.head()

df_x = pd.merge(df_customer_revenue_by_year,df_active_card_count_by_customer, how='left', on=['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR'])
print(len(df_x))

df_customer_fleet_size = df_active_card_count_by_customer.groupby(['CUSTOMER_ID','CUSTOMER']).ACTIVE_CARD_COUNT.max().reset_index()
df_customer_fleet_size.columns = ['CUSTOMER_ID', 'CUSTOMER', 'FLEET_SIZE']

df_x = pd.merge(df_x, df_customer_fleet_size, how='left', on=['CUSTOMER_ID','CUSTOMER'])
print(len(df_x))

df_x = pd.merge(df_x, df_customer_MRU, how='left', on=['CUSTOMER_ID','CUSTOMER'])
print(len(df_x))

df_x = pd.merge(df_x, df_customer_program, how='left', on=['CUSTOMER_ID','CUSTOMER'])
print(len(df_x))

df_x.loc[df_x.CUSTOMER_BUSINESS_PROGRAM_NAME=='WEX Universal','MRU'] = 4120
df_x.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
revenue_by_MRU = df_x.groupby(['MRU','REVENUE_YEAR']).REVENUE_AMOUNT_USD.sum().reset_index()
revenue_by_MRU[revenue_by_MRU.REVENUE_YEAR.isin([2020,2021])]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
revenue_by_MRU = df_x.groupby(['MRU','REVENUE_YEAR']).ACTIVE_CARD_COUNT.sum().reset_index()
revenue_by_MRU[revenue_by_MRU.REVENUE_YEAR.isin([2020,2021])]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# create dataset by year,customer,business program, mru
# join in customer fleet size
# render revenue by MRU for 2020
# render revenue by MRU for 2021
# find the fleet size threshold that will move customers in oder to right size the revenue amounts
# summarize the counts

#df_active_card_count_by_customer = df_a.groupby(['CUSTOMER_ID','CUSTOMER','YEAR_NUMBER']).ACTIVE_CARD_COUNT.sum().reset_index()
#df_active_card_count_by_customer.columns = ['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR','ACTIVE_CARD_COUNT']
#print(len(df_active_card_count_by_customer))

#df_customer_fleet_size = df_active_card_count_by_customer.groupby(['CUSTOMER_ID','CUSTOMER']).ACTIVE_CARD_COUNT.max().reset_index()
#df_customer_fleet_size.columns = ['CUSTOMER_ID', 'CUSTOMER', 'FLEET_SIZE']

#df_active_card_count_by_customer = pd.merge(df_active_card_count_by_customer,df_customer_fleet_size, on=['CUSTOMER_ID','CUSTOMER'], how='left')
#print(len(df_active_card_count_by_customer))

#df_customer_program = pd.merge(df_customer_program,df_active_card_count_by_customer, how='left', on=['CUSTOMER_ID','CUSTOMER','REVENUE_YEAR'])
#print(len(df_customer_program))
#df_customer_program.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df_active_card_count_by_customer))
df_active_card_count_by_customer.drop_duplicates(subset=['CUSTOMER_ID'], inplace=True)
print(len(df_active_card_count_by_customer))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#COST_BY_SEGMENT_df = ... # Compute a Pandas dataframe to write into COST_BY_SEGMENT

# Write recipe outputs
#COST_BY_SEGMENT = dataiku.Dataset("COST_BY_SEGMENT")
#COST_BY_SEGMENT.write_with_schema(COST_BY_SEGMENT_df)
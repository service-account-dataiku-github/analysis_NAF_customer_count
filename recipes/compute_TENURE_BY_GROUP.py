# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import matplotlib.pyplot as plt

# ***************************************
#
# This function calculates and observes retention rates for 2019-2023
# and then forecasts retention rates for 2023-2035
# and then uses the 2019-2035 retention rates to estimate EXEPECTED/AVG RETENTION
#
# Input Data Set to this function:
# NAFCUSTOMER_REVENUE_AGGREGATED
# Each row represents one customer per year/month
# dimensions: customer name, setup date, BI_CUS_PAR, BI_PRODUCT, BI_MRU
# measures: revenue, spend, gallons, active cards and outstanding cards
#
# **************************************
#
# The data going into this step is defined in the follopwing Input View:
#
#  CREATE VIEW DATAIKU_ANALYTICS_SUB.DATA_SCIENCE.NAFCUSTOMER_REVENUE_AGGREGATED
#  AS
#  SELECT
#  CUSTOMER_ID,
#  CUSTOMER,
#  YEAR(REVENUE_DATE) as REVENUE_YEAR,
#  MONTH(REVENUE_DATE) as REVENUE_MONTH,
#  MIN(SETUP_DATE) AS SETUP_DATE,
#  MIN(BI_CUS_PAR) as BI_CUS_PAR,
#  MIN(BI_PRODUCT) as BI_PRODUCT,
#  MIN(BI_MRU) as BI_MRU,
#  SUM(REVENUE_AMOUNT_USD) as REVENUE_AMOUNT_USD,
#  SUM(GROSS_SPEND_AMOUNT) as GROSS_SPEND_AMOUNT,
#  SUM(PURCHASE_GALLONS_QTY) PURCHASE_GALLONS_QTY,
#  SUM(TOTAL_ACTIVE_CARD_COUNT) as TOTAL_ACTIVE_CARD_COUNT
#  FROM DATAIKU_ANALYTICS_SUB.DATA_SCIENCE.NAFCUSTOMER_REVENUE
#  WHERE (FUNDED_BY IS NULL OR FUNDED_BY='Unknown' OR FUNDED_BY='WEX Funded')
#  AND (NOT((REVENUE_CODE='26') AND (CUSTOMER_ID IS NULL)))
#  AND (NOT((REVENUE_CODE='E8') AND (CUSTOMER_ID IS NULL)))
#  GROUP BY 1,2,3,4
#
# ***************************************

from datetime import date, datetime, timedelta
import time

t0 = time.time()

# Read recipe inputs
NAFCUSTOMER_REVENUE_AGGREGATED = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED")
NAFCUSTOMER_REVENUE_AGGREGATED_df = NAFCUSTOMER_REVENUE_AGGREGATED.get_dataframe()

t1 = time.time()
print("load duration", (t1-t0)/60.0, "minutes")

print(len(NAFCUSTOMER_REVENUE_AGGREGATED_df))
NAFCUSTOMER_REVENUE_AGGREGATED_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# create a copy of the input dataset
# this leaves the original dataset in memory and allows for a quick reload without having to back to Snowflake

print(len(NAFCUSTOMER_REVENUE_AGGREGATED_df), "rows in NAFCUSTOMER_REVENUE_AGGREGATED")
df = NAFCUSTOMER_REVENUE_AGGREGATED_df.copy()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Prep data
#
# cast year to integer and create a revenue date column using month and year
#

df['CUSTOMER_ID'] = df['CUSTOMER_ID'].astype('Int64')
df['SETUP_DATE_DT'] = pd.to_datetime(df['SETUP_DATE'])
df['SETUP_YEAR'] = df.SETUP_DATE_DT.dt.year

df['REVENUE_YEAR'] = df['REVENUE_YEAR'].astype('Int64')
df['REVENUE_DATE'] = df['REVENUE_MONTH'].astype(str) + '-' + df['REVENUE_YEAR'].astype(str)
df['REVENUE_DATE'] = pd.to_datetime(df['REVENUE_DATE'], format='%m-%Y').dt.strftime('%m-%Y')

#
# filter out Jan 2023 as it represents only a partial year
#

print(len(df), "before filter")
df = df[df.REVENUE_YEAR!=2023]
print(len(df), "after filter, removing 2023")

# break down of revenue_year
df.REVENUE_YEAR.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.CUSTOMER_ID.nunique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# plot out sum of revenue per year
# in order to do a quick reonciliation against 

# aggregate revenue at annual grain
df_revenue_per_year = df.groupby(['REVENUE_YEAR']).REVENUE_AMOUNT_USD.sum().reset_index()
df_revenue_per_year.REVENUE_AMOUNT_USD = df_revenue_per_year.REVENUE_AMOUNT_USD/1000000
df_revenue_per_year.head()

max_revenue = df_revenue_per_year.REVENUE_AMOUNT_USD.max()

chart_revenue_year = [] 
for y in df_revenue_per_year.REVENUE_YEAR.tolist():
    chart_revenue_year.append(str(y))
    
fig, ax1 = plt.subplots(figsize=(8,3))
ax1.plot(chart_revenue_year,df_revenue_per_year['REVENUE_AMOUNT_USD'], marker='o')
ax1.set_xlabel('YEAR', fontsize=14)
ax1.set_ylabel('REVENUE (M)', fontsize=14)
ax1.grid()
ax1.set_ylim(ymin=0, ymax=max_revenue*1.15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
fig.autofmt_xdate()
#ax1.xaxis.set_major_locator(mdates.YearLocator())
plt.show()

df_revenue_per_year.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Fleet Sizes
# zero cards : 0
# small      : 1-20
# medium     : 21-114
# large      : 115-1700
# X-Large    : > 1700

df_monthly_card_count = df.groupby(['CUSTOMER_ID','REVENUE_DATE']).TOTAL_ACTIVE_CARD_COUNT.sum().reset_index()
df_max_monthly_card_count = df_monthly_card_count.groupby(['CUSTOMER_ID']).TOTAL_ACTIVE_CARD_COUNT.max().reset_index()

df_max_monthly_card_count.TOTAL_ACTIVE_CARD_COUNT = df_max_monthly_card_count.TOTAL_ACTIVE_CARD_COUNT.astype('Int64')

df_max_monthly_card_count['FLEET_SIZE'] = 'not set'
df_max_monthly_card_count.loc[df_max_monthly_card_count.TOTAL_ACTIVE_CARD_COUNT<=0,'FLEET_SIZE'] = 'zero cards'
df_max_monthly_card_count.loc[df_max_monthly_card_count.TOTAL_ACTIVE_CARD_COUNT.between(1,20),'FLEET_SIZE'] = 'NAF Small Fleet (<=20 cards)'
df_max_monthly_card_count.loc[df_max_monthly_card_count.TOTAL_ACTIVE_CARD_COUNT>20,'FLEET_SIZE'] = 'NAF Large Fleet (>20 cards)'

df_max_monthly_card_count['FLEET_SIZE_GRANULAR'] = 'not set'
df_max_monthly_card_count.loc[df_max_monthly_card_count.TOTAL_ACTIVE_CARD_COUNT<=0,'FLEET_SIZE_GRANULAR'] = 'zero cards'
for i in range(1,21):
    df_max_monthly_card_count.loc[df_max_monthly_card_count.TOTAL_ACTIVE_CARD_COUNT==i,'FLEET_SIZE_GRANULAR'] = str(i) + ' card(s)'
df_max_monthly_card_count.loc[df_max_monthly_card_count.TOTAL_ACTIVE_CARD_COUNT>20,'FLEET_SIZE_GRANULAR'] = '> 20 cards'
    
df_max_monthly_card_count.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_max_monthly_card_count.FLEET_SIZE_GRANULAR.value_counts()
df_max_monthly_card_count.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_customer_min_setup = df.groupby(['CUSTOMER_ID']).SETUP_YEAR.min().reset_index()
df_customer_min_setup.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_fleet_size_customer_count_granular = df_max_monthly_card_count.groupby(['FLEET_SIZE_GRANULAR']).CUSTOMER_ID.nunique().reset_index()
df_fleet_size_customer_count_granular.columns = ['FLEET_SIZE_GRANULAR', 'CUSTOMER_COUNT']
df_fleet_size_customer_count_granular.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_fleet_size_customer_count = df_max_monthly_card_count.groupby(['FLEET_SIZE']).CUSTOMER_ID.nunique().reset_index()
df_fleet_size_customer_count.columns = ['FLEET_SIZE', 'CUSTOMER_COUNT']
df_fleet_size_customer_count.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_fleet_size_card_count = df_max_monthly_card_count.groupby(['FLEET_SIZE']).TOTAL_ACTIVE_CARD_COUNT.sum().reset_index()
df_fleet_size_card_count.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df))
df_activity = df[['CUSTOMER_ID','REVENUE_YEAR','REVENUE_DATE','PURCHASE_GALLONS_QTY','GROSS_SPEND_AMOUNT','REVENUE_AMOUNT_USD']].copy()
print(len(df_activity))

# drop nulls -- todo: figure out what these few null rows represent
df_activity = df_activity[~df_activity.CUSTOMER_ID.isnull()]
print(len(df_activity))

df_fleet_size = df_max_monthly_card_count[['CUSTOMER_ID','FLEET_SIZE','FLEET_SIZE_GRANULAR']].copy()
df_fleet_size['CUSTOMER_ID'] = df_fleet_size['CUSTOMER_ID'].astype('Int64')

print(len(df_activity))
df_activity = pd.merge(df_activity, df_fleet_size, on='CUSTOMER_ID', how='left')
print(len(df_activity))

df_customer_min_setup['CUSTOMER_ID'] = df_customer_min_setup['CUSTOMER_ID'].astype('Int64')
df_activity = pd.merge(df_activity, df_customer_min_setup, on='CUSTOMER_ID', how='left')
print(len(df_activity))

df_activity.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_max_monthly_card_count.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_revenue_per_year = df_activity.groupby(['REVENUE_YEAR','FLEET_SIZE']).REVENUE_AMOUNT_USD.sum().reset_index()
df_revenue_per_year.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_spend_per_year = df_activity.groupby(['REVENUE_YEAR','FLEET_SIZE']).GROSS_SPEND_AMOUNT.sum().reset_index()
df_spend_per_year.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_revenue_per_year.FLEET_SIZE.unique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
max_spend = df_spend_per_year.GROSS_SPEND_AMOUNT.max()
    
fig, ax1 = plt.subplots(figsize=(8,3))

df_spend_per_year['REVENUE_YEAR'] = df_spend_per_year['REVENUE_YEAR'].astype(str)

# s = df_revenue_per_year.CUSTOMER_FLEET_SIZE.unique()
# the following is this same list rearranged
groups = ['NAF Large Fleet (>20 cards)', 'NAF Small Fleet (<=20 cards)','zero cards']

legend_items = []
for s in groups:
    ax1.plot(df_spend_per_year[df_spend_per_year.FLEET_SIZE==s].REVENUE_YEAR,
             df_spend_per_year[df_spend_per_year.FLEET_SIZE==s].GROSS_SPEND_AMOUNT, marker='o')
    legend_items.append(s)
    
ax1.set_xlabel('YEAR', fontsize=14)
ax1.set_ylabel('SPEND ($)', fontsize=14)
ax1.grid()
ax1.set_ylim(ymin=0, ymax=max_spend*1.15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
fig.autofmt_xdate()
plt.title('NAF Spend by Fleet Size')
plt.legend(legend_items, bbox_to_anchor=(1.05,1.0), loc='upper left')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
max_revenue = df_revenue_per_year.REVENUE_AMOUNT_USD.max()
    
fig, ax1 = plt.subplots(figsize=(8,3))

df_revenue_per_year['REVENUE_YEAR'] = df_revenue_per_year['REVENUE_YEAR'].astype(str)

# s = df_revenue_per_year.CUSTOMER_FLEET_SIZE.unique()
# the following is this same list rearranged so that the legend is ordered correctly
groups = ['NAF Large Fleet (>20 cards)', 'NAF Small Fleet (<=20 cards)','zero cards']

legend_items = []
for s in groups:
    ax1.plot(df_revenue_per_year[df_revenue_per_year.FLEET_SIZE==s].REVENUE_YEAR,
             df_revenue_per_year[df_revenue_per_year.FLEET_SIZE==s].REVENUE_AMOUNT_USD, marker='o')
    legend_items.append(s)
    
ax1.set_xlabel('YEAR', fontsize=14)
ax1.set_ylabel('REVENUE ($)', fontsize=14)
ax1.grid()
ax1.set_ylim(ymin=0, ymax=max_revenue*1.15)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
fig.autofmt_xdate()
plt.title('NAF Revenue by Fleet Size')
plt.legend(legend_items, bbox_to_anchor=(1.05,1.0), loc='upper left')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_activity.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
groups = ['NAF Large Fleet (>20 cards)', 'NAF Small Fleet (<=20 cards)','zero cards']
filter_group = 'NAF Large Fleet (>20 cards)'

_fleet_size = []
_observed_avg_retention_rate = []
_expected_tenure = []
_revenue_year = []
_observed_customer_count = []
_observed_customer_percent = []
_chart_forecast_year = []
_chart_forecast_from_first = []

removed = 'zero cards'
groups = ['1 card(s)', '2 card(s)','3 card(s)','4 card(s)', 
          '5 card(s)', '6 card(s)','7 card(s)', '8 card(s)', '9 card(s)', '10 card(s)', 
          '11 card(s)', '12 card(s)', '13 card(s)','14 card(s)','15 card(s)', 
          '16 card(s)','17 card(s)','18 card(s)','19 card(s)','20 card(s)','> 20 cards']

for granular_filter_group in groups:

    #filter_group = 'NAF Small Fleet (<=20 cards)'

    df_sub = df_activity[df_activity.SETUP_YEAR<2019]
    #df_sub = df_sub[df_sub.FLEET_SIZE==filter_group]
    df_sub = df_sub[df_sub.FLEET_SIZE_GRANULAR==granular_filter_group]

    df_sub = df_sub.groupby(['REVENUE_YEAR']).CUSTOMER_ID.nunique().reset_index()
    df_sub.columns = ['REVENUE_YEAR','CUSTOMER_COUNT']
    df_sub.REVENUE_YEAR = df_sub.REVENUE_YEAR.astype(str)
    df_sub.CUSTOMER_COUNT = df_sub.CUSTOMER_COUNT.astype('Int64')

    df_sub['SURVIVED_FROM_FIRST'] = (df_sub.CUSTOMER_COUNT / df_sub.iloc[0].CUSTOMER_COUNT)
    df_sub['SURVIVED_FROM_FIRST_PERCENT'] = (df_sub.CUSTOMER_COUNT / df_sub.iloc[0].CUSTOMER_COUNT)*100
    df_sub['CUSTOMERS_LEFT'] = 0

    row_count = len(df_sub)

    row_index = 0
    retention_observations = []
    observed_retentions = 0
    for i in range(row_count):
        if i>0:
            survived = df_sub.iloc[i].CUSTOMER_COUNT / df_sub.iloc[i-1].CUSTOMER_COUNT
            df_sub.loc[i,'SURVIVED_FROM_PREV'] = survived
            df_sub.loc[i,'CUSTOMERS_LEFT'] = df_sub.iloc[i-1].CUSTOMER_COUNT - df_sub.iloc[i].CUSTOMER_COUNT
            retention_observations.append(survived)
        row_index+=1

    avg_retention_rate = sum(retention_observations)/len(retention_observations)                
    print("----")
    print("Observed Average Retention Rate (2019-2022):", 100*round(avg_retention_rate,3), "%")
    print(retention_observations)
    len(retention_observations)                

    forecast_year = 2022

    df_sub['FORECAST_CUSTOMER_COUNT'] = np.nan
    df_sub['FORECAST_FROM_FIRST_PERCENT'] = np.nan

    forecast_years = 15

    for i in range(forecast_years+1):

        forecast_year += 1
        if i==0:
            forecast_customer_count = round(avg_retention_rate * df_sub.iloc[i+row_count-1].CUSTOMER_COUNT,0)
            customers_left = df_sub.iloc[i+row_count-1].CUSTOMER_COUNT - forecast_customer_count
        else:
            forecast_customer_count = round(avg_retention_rate * df_sub.iloc[i+row_count-1].FORECAST_CUSTOMER_COUNT,0)
            customers_left = df_sub.iloc[i+row_count-1].FORECAST_CUSTOMER_COUNT - forecast_customer_count

        forecast_from_first_percent = (forecast_customer_count/df_sub.iloc[0].CUSTOMER_COUNT)*100

        dict = {'REVENUE_YEAR':forecast_year,'CUSTOMERS_LEFT': customers_left, 'SURVIVED_FROM_PREV': avg_retention_rate, 'FORECAST_CUSTOMER_COUNT':forecast_customer_count,'FORECAST_FROM_FIRST_PERCENT':forecast_from_first_percent}
        df_sub = df_sub.append(dict, ignore_index = True)


    df_sub['CUSTOMERS_LEFT_PERCENT'] = df_sub['CUSTOMERS_LEFT'] / df_sub.iloc[0].CUSTOMER_COUNT
    df_sub.FORECAST_CUSTOMER_COUNT = df_sub.FORECAST_CUSTOMER_COUNT.astype('Int64') 

    chart_revenue_year = df_sub.REVENUE_YEAR[0:row_index].tolist()
    chart_customer_count = df_sub.CUSTOMER_COUNT[0:row_index].tolist()

    chart_survived_from_first = df_sub.SURVIVED_FROM_FIRST_PERCENT[0:row_index].tolist()

    l = []
    max_i = 0
    for i in df_sub.REVENUE_YEAR[row_index:].tolist():
        l.append(str(int(i)))
        max_i = i

    chart_forecast_year = l
    chart_forecast_customer_count = df_sub.FORECAST_CUSTOMER_COUNT[row_index:].tolist()
    chart_forecast_from_first = df_sub.FORECAST_FROM_FIRST_PERCENT[row_index:].tolist()

    max_y = df_sub.CUSTOMER_COUNT.max()
    max_y = max_y + max_y*0.1

    _revenue_year.append(chart_revenue_year)
    _observed_customer_count.append(chart_customer_count)    
    
    fig, ax1 = plt.subplots(figsize=(10,5))
    plt.plot(chart_revenue_year,chart_customer_count, marker='o', c='C0')
    plt.plot(chart_forecast_year,chart_forecast_customer_count, marker='o', c='black', linestyle='dashed')
    ax1.set_ylim(ymin=0, ymax=max_y)
    ax1.set_xlabel('YEAR', fontsize=14)
    #ax1.set_xlim(xmin=2019, xmax=2023+forecast_years)
    ax1.set_ylabel('CUSTOMER COUNT', fontsize=14)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.title(' Count of Retained Customers, ' + granular_filter_group)
    plt.legend(['observed','forecasted'])

    _observed_customer_percent.append(chart_survived_from_first)
    _chart_forecast_year.append(chart_forecast_year)
    _chart_forecast_from_first.append(chart_survived_from_first)
    
    fig, ax2 = plt.subplots(figsize=(10,5))
    plt.plot(chart_revenue_year,chart_survived_from_first, marker='o', c='C0')
    plt.plot(chart_forecast_year,chart_forecast_from_first, marker='o', c='black', linestyle='dashed')
    ax2.set_ylim(ymin=0, ymax=110)
    ax2.set_xlabel('YEAR', fontsize=14)
    ax2.set_ylabel('Percent Survived (%)', fontsize=14)
    plt.title(' % of Retained Customers, ' + granular_filter_group)
    plt.legend(['observed','forecasted'])
    #plt.subplots_adjust(left=2.0,right=3.0)
    plt.show()

    fraction_remaining = 1
    avg_tenure = 0
    check_sum = 0
    for i in range(forecast_years):

        avg_tenure += df_sub.iloc[i].CUSTOMERS_LEFT_PERCENT * i

        fraction_remaining -=  df_sub.iloc[i].CUSTOMERS_LEFT_PERCENT
        check_sum += df_sub.iloc[i].CUSTOMERS_LEFT_PERCENT

    check_sum += fraction_remaining

    avg_tenure += fraction_remaining * i
    print("forecast period:", forecast_years)
    print("AVG Tenure (Observed + Forecasted):", round(avg_tenure,2), "Years")
    print("check_sum:", round(check_sum,2))
    
    _fleet_size.append(granular_filter_group)
    _observed_avg_retention_rate.append(avg_retention_rate)
    _expected_tenure.append(avg_tenure)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

fig, ax2 = plt.subplots(figsize=(10,6.5))

color_step_size = int(255 / len(_fleet_size))
color_step_size = 30

current_color_step_size = 255
for i in range(len(_fleet_size)):
    
    color = "#" + rgb_to_hex((current_color_step_size,0,0))
    current_color_step_size -= color_step_size
    if current_color_step_size<=0:
        current_color_step_size = 0

    fleet_size = _fleet_size[i]
    observed_avg_retention_rate = _observed_avg_retention_rate[i]
    expected_tenure = _expected_tenure[i]
    revenue_year = _revenue_year[i]
    observed_customer_percent = _observed_customer_percent[i]
    chart_forecast_year = _chart_forecast_year[i]
    chart_forecast_from_first = _chart_forecast_from_first[i]
    
    plt.plot(revenue_year,observed_customer_percent, marker='o', color = color)
    #plt.plot(chart_forecast_year,chart_forecast_from_first, marker='o', c='black', linestyle='dashed')
    
ax2.set_ylim(ymin=0, ymax=103)
ax2.set_xlabel('YEAR', fontsize=14)
ax2.set_ylabel('Percent Survived (%)', fontsize=14)
ax2.grid()
plt.title(' % of Retained Customers by Fleet Size')
plt.legend(_fleet_size, bbox_to_anchor=(1.05,1.0), loc='upper left')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
_color_list = []
step_size=30
for i in _fleet_size_num:
    red_val = 255-i*step_size
    if red_val<0:
        red_val=0
    _color_list.append("#" + rgb_to_hex((red_val,0,0)))
print(_color_list)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
_fleet_size_num = []
for i in range(len(_fleet_size)):
    _fleet_size_num.append(i)
_fleet_size_num

fig, ax2 = plt.subplots(figsize=(4.5,2.5))
ax2.set_xlabel('Fleet Size (Number of Cards)', fontsize=14)
ax2.set_ylabel('Avg Retention (%)', fontsize=14)
ax2.grid()

plt.scatter(_fleet_size_num, _observed_avg_retention_rate, s=15, c=_color_list)
plt.title('Fleet Size vs Avg Retention Rate')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
fig, ax2 = plt.subplots(figsize=(4.5,2.5))
ax2.set_xlabel('Fleet Size (Number of Cards)', fontsize=14)
ax2.set_ylabel('Expected Tenure (Years)', fontsize=14)
ax2.grid()

plt.scatter(_fleet_size_num, _expected_tenure, s=15,c=_color_list)
plt.title('Fleet Size vs Expected Tenure')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
_fleet_size_num

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
retention_observations

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(_fleet_size)
print()
print(_observed_avg_retention_rate)
print()
print(_expected_tenure)
print()
print(_revenue_year)
print()
print(_observed_customer_count)
print()
print(_observed_customer_percent)
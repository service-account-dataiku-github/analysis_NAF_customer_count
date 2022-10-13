# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import pickle
from dateutil.relativedelta import relativedelta
import gc
from re import finditer

## Find DD DU
from helper import preprocess_data
from patterns import find_drawdowns, find_drawups

## MATCHING
import name_matching
from name_matching import name_match
import transaction_matching
from transaction_matching import transaction_match

## CONSOLIDATION
from consolidation import combine_matches, consolidate_matches, find_attritions, find_new_accounts, get_attrition_status, get_new_account_status

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
start_date = dataiku.get_custom_variables()['start_date']
end_date = dataiku.get_custom_variables()['end_date']

consistency = int(dataiku.get_custom_variables()['consistency'])
drawdown_period_average = int(dataiku.get_custom_variables()['drawdown_period_average'])
drawdown = int(dataiku.get_custom_variables()['drawdown'])
drawdown_fwd_check = int(dataiku.get_custom_variables()['drawdown_fwd_check'])
drawdown_lookback_period = int(dataiku.get_custom_variables()['drawdown_lookback_period'])
drawup_lookfwd_period = int(dataiku.get_custom_variables()['drawup_lookfwd_period'])
statistics_period = int(dataiku.get_custom_variables()['statistics_period'])
inactive_period = int(dataiku.get_custom_variables()['inactive_period'])

## MATCHING VARIABLES
month_diff_h = int(dataiku.get_custom_variables()['month_diff_h'])
month_diff_l = int(dataiku.get_custom_variables()['month_diff_l'])
sd_mul = int(dataiku.get_custom_variables()['sd_mul'])
max_city_distance = int(dataiku.get_custom_variables()['max_city_distance'])
threshold_score_step1 = int(dataiku.get_custom_variables()['threshold_score_step1'])
threshold_score_step2 = int(dataiku.get_custom_variables()['threshold_score_step2'])

## RUN TYPE
run = dataiku.get_custom_variables()['run_type']

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def date_tz_naive(pd_s):
    return pd.to_datetime(pd_s).apply(lambda x:x.tz_localize(None))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
NAFCUSTOMER_C360_ACCOUNTS = dataiku.Dataset("NAFCUSTOMER_C360_ACCOUNTS")
NAFCUSTOMER_C360_ACCOUNTS_df = NAFCUSTOMER_C360_ACCOUNTS.get_dataframe()
print(len(NAFCUSTOMER_C360_ACCOUNTS_df))

NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED = dataiku.Dataset("NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED")
NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df = NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED.get_dataframe()
print(len(NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_v = NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df

print(len(df_v))
df_v['REVENUE_DATE'] = df_v.REVENUE_MONTH.astype(str) + "/01/" + df_v.REVENUE_YEAR.astype(str)
df_v['REVENUE_DATE'] = date_tz_naive(df_v['REVENUE_DATE'])
print(len(df_v))
df_v.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df_v))
df_v = df_v[df_v['REVENUE_DATE'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))].copy()
df_v = df_v.dropna(subset=['CUSTOMER_ACCOUNT_ID'])
df_v = df_v[df_v['CUSTOMER_SOURCE_SYSTEM_CODE'].isin(['TANDEM', 'SIEBEL'])]
print(len(df_v))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_v['CUSTOMER_ACCOUNT_ID'] = df_v['CUSTOMER_ACCOUNT_ID'].astype('int64')
df_v['REVENUE_DATE'] = pd.to_datetime(df_v['REVENUE_DATE'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
states = list(df_v['ACCOUNT_STATE'].unique())
states_dict = {s:s.upper() for s in states}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_v['ACCOUNT_STATE'] = df_v['ACCOUNT_STATE'].map(states_dict)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## remove the unneccesary columns
remove_cols=['REVENUE_MONTH','REVENUE_YEAR', 'REVENUE_QUARTER']
df_v = df_v.drop([x for x in remove_cols if x in df_v.columns], axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_v.sort_values(['REVENUE_DATE'], inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
seen_accounts = df_v[df_v['PURCHASE_GALLONS_QTY'] > 0].groupby(['CUSTOMER_ACCOUNT_ID'], as_index=False)[['REVENUE_DATE']].first()
seen_accounts['FIRST_DATE'] = seen_accounts['REVENUE_DATE'] - pd.DateOffset(months=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_v.REVENUE_DATE.value_counts(dropna=False)
print(len(df_v))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from helper import *

#---------------------
# input vars
df = df_v
period_end_date = end_date
match_type = 'program_flip'
period_start_date=None
split=None
#------------------------

drawdown = (100 - drawdown)/100
drawdown_fwd_check /= 100

inactive_date_start = pd.to_datetime(period_end_date) + relativedelta(months=-inactive_period)

if match_type == 'conversion':
    df = df[df['CUSTOMER_SOURCE_SYSTEM_CODE'] == 'TANDEM'].copy()

df = df[df['REVENUE_DATE'] <= period_end_date].copy()

if period_start_date:
    period_start_date = pd.to_datetime(period_start_date)
    df = df[df['REVENUE_DATE'] >= period_start_date].copy()

all_account_ids = list(df['CUSTOMER_ACCOUNT_ID'].unique())

if not split:
    split=1

all_account_ids_n = list(split_list(all_account_ids, split))

drop_df = pd.DataFrame()

for sublist in tqdm(all_account_ids_n):

    dd_find = df[df['CUSTOMER_ACCOUNT_ID'].isin(sublist)].copy()

    ## Find consistent customers
    consistent_customers_dd = find_consistent_customers(dd_find, consecutive=consistency)
    if len(consistent_customers_dd) == 0:
        continue

    dd_find = dd_find[dd_find['CUSTOMER_ACCOUNT_ID'].isin(consistent_customers_dd)].copy()

    ## Add padding, find the n months average and compute drawdown indicator based on the rules
    dd_find = add_padding(dd_find, padding=statistics_period, last_date=period_end_date)
    dd_find = find_average(dd_find, n=statistics_period)

    dd_find['DD_INDICATOR'] = np.where(((drawdown*(dd_find['LAST_N_MONTHS_AVG'].round(3)) >
                                         dd_find['PURCHASE_GALLONS_QTY'].round(3)) &
                                        (dd_find['NEXT_N_MONTHS_AVG'].round(3) <
                                         drawdown_fwd_check*dd_find['LAST_N_MONTHS_AVG'].round(3))),
                                       True, False)

    ## Find the first drawdown and also the list of customers
    pflip_dd = dd_find[dd_find['DD_INDICATOR'] == True].copy()
    pflip_dd.drop_duplicates('CUSTOMER_ACCOUNT_ID', inplace=True)
    first_drop_idx = pflip_dd.index
    pflip_dd_customers = list(dd_find['CUSTOMER_ACCOUNT_ID'].unique())
    first_drop = dd_find.iloc[first_drop_idx]

    ## Identify the lookback period
    first_drop = first_drop[['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE']].copy()
    first_drop = first_drop[first_drop['REVENUE_DATE'] <= inactive_date_start].copy()
    first_drop['START_DATE']  = first_drop['REVENUE_DATE'] - pd.DateOffset(months=drawdown_lookback_period)
    first_drop.rename(columns = {'REVENUE_DATE':'DD_DATE'}, inplace=True)
    dd_find_df = dd_find[dd_find['CUSTOMER_ACCOUNT_ID'].isin(pflip_dd_customers)]
    dd_find_df = dd_find_df.merge(first_drop, on=['CUSTOMER_ACCOUNT_ID'])
    dd_find_df = dd_find_df[dd_find_df['REVENUE_DATE'].between(dd_find_df['START_DATE'],dd_find_df['DD_DATE'])].copy()

    ## Compute the sharpest fall from the lookback period
    dd_find_df.sort_values(['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE'], inplace=True)
    dd_find_df['DROP'] = dd_find_df.groupby(['CUSTOMER_ACCOUNT_ID'])['PURCHASE_GALLONS_QTY'].diff(-1)

    ## Find the corresponding period and remove duplicates in case of a similar values
    drop_idx = dd_find_df.groupby(['CUSTOMER_ACCOUNT_ID'])['DROP'].transform(max) == dd_find_df['DROP']
    drop_month_df = dd_find_df[drop_idx].copy()
    drop_month_df.drop_duplicates(['CUSTOMER_ACCOUNT_ID'], inplace=True)

    ## remove the first record
    dd_find = dd_find.groupby('CUSTOMER_ACCOUNT_ID').apply(lambda group: group.iloc[1:, 1:]).reset_index()
    dd_find.drop('level_1', axis=1, inplace=True)

    ## Find the time periods for calculating statistics (mean and standard deviation)
    drop_month_df.rename(columns = {'REVENUE_DATE':'DROP_DATE'}, inplace=True)
    dd_find = dd_find.merge(drop_month_df[['CUSTOMER_ACCOUNT_ID', 'DROP_DATE']], on='CUSTOMER_ACCOUNT_ID')
    dd_find['END_DATE'] = dd_find['DROP_DATE'] - pd.DateOffset(months=3)
    dd_find['START_DATE'] = dd_find['END_DATE'] - pd.DateOffset(months=statistics_period-1)
    pflip_12_data = dd_find[dd_find['REVENUE_DATE'].between(dd_find['START_DATE'], dd_find['END_DATE'])].copy()

    ## Calculate Mean and Standard Deviation
    dd_stat = pflip_12_data.groupby(['CUSTOMER_ACCOUNT_ID'], as_index=False).agg({'PURCHASE_GALLONS_QTY':['mean','std']})
    dd_stat.columns = ['CUSTOMER_ACCOUNT_ID_DD', 'MEAN_DD','STD_DD']
    drop_month_df = drop_month_df.merge(dd_stat,
                                        left_on='CUSTOMER_ACCOUNT_ID',
                                        right_on='CUSTOMER_ACCOUNT_ID_DD',
                                        how='left')

    drop_df = pd.concat([drop_df, drop_month_df], ignore_index=True)

drop_df.drop(['CUSTOMER_ACCOUNT_ID_DD'], axis=1, inplace=True)
drop_df.rename(columns={'DROP_DATE':'DRAW_DOWN_DATE',
                        'DROP':'DROP_QTY'}, inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(drop_df.CUSTOMER_ACCOUNT_ID.unique())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
drop_df = drop_df[['CUSTOMER_ACCOUNT_ID','DRAW_DROP_DATE','DROP_QTY','MEAN_DD','STD_DD']]
drop_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

CALCULATED_DRAW_DOWNS_df = drop_df

# Write recipe outputs
CALCULATED_DRAW_DOWNS = dataiku.Dataset("CALCULATED_DRAW_DOWNS")
CALCULATED_DRAW_DOWNS.write_with_schema(CALCULATED_DRAW_DOWNS_df)
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.



#CALCULATED_DRAW_DOWNS_df = NAFCUSTOMER_C360_ACCOUNTS_df # For this sample code, simply copy input to output


# Write recipe outputs
#CALCULATED_DRAW_DOWNS = dataiku.Dataset("CALCULATED_DRAW_DOWNS")
#CALCULATED_DRAW_DOWNS.write_with_schema(CALCULATED_DRAW_DOWNS_df)
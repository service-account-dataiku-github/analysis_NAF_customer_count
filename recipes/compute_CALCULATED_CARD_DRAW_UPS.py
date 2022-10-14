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
# Read recipe inputs
NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED = dataiku.Dataset("NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED")
NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df = NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED.get_dataframe()

print(len(NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df))
print(len(NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df.CUSTOMER.unique()))
NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_v = NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df
print(len(df_v))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def date_tz_naive(pd_s):
    return pd.to_datetime(pd_s).apply(lambda x:x.tz_localize(None))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df_v))
df_v['REVENUE_DATE'] = df_v.REVENUE_MONTH.astype(str) + "/01/" + df_v.REVENUE_YEAR.astype(str)
df_v['REVENUE_DATE'] = date_tz_naive(df_v['REVENUE_DATE'])
print(len(df_v))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df_v))
df_v = df_v[df_v['REVENUE_DATE'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))].copy()
df_v = df_v.dropna(subset=['CUSTOMER'])
print(len(df_v))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_v['REVENUE_DATE'] = pd.to_datetime(df_v['REVENUE_DATE'])

df_v = df_v[['CUSTOMER','REVENUE_DATE', 'ACTIVE_CARD_COUNT']]

df_cust =  df_v.groupby(by=["CUSTOMER","REVENUE_DATE"]).sum().reset_index()
df_cust.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from helper import *

def split_list(lst, n):
    '''
    Splits a list into almost equal n parts
    '''
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
match_type = "program_flip"
period_start_date = start_date
period_end_date = None
drawup_window = drawup_lookfwd_period
statistics_period = statistics_period
split = None

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
period_start_date = pd.to_datetime(period_start_date)
df_v = df_v[df_v['REVENUE_DATE'] >= period_start_date].copy()

if period_end_date:
    period_end_date = pd.to_datetime(period_end_date)
    df_v = df_v[df_v['revenue_date'] <= period_end_date].copy()

all_account_ids = list(df_v['CUSTOMER'].unique())

if not split:
    split=1

all_account_ids_n = list(split_list(all_account_ids, split))

rise_df = pd.DataFrame()

for sublist in tqdm(all_account_ids_n):

    du_find = df_v[df_v['CUSTOMER'].isin(sublist)].copy()

    ## Filter Non-Zero Records and find the first non zero transaction date
    du_find = du_find[du_find['ACTIVE_CARD_COUNT'] > 0]
    
    du_find.sort_values(['REVENUE_DATE'], inplace=True)

    du_agg = du_find.groupby(['CUSTOMER'], as_index=False)[['REVENUE_DATE']].min()
    
    du_agg['DU_INDICATOR'] = np.where((du_agg['REVENUE_DATE'] > period_start_date), True, False)
    du_agg.rename(columns={'REVENUE_DATE':'DU_DATE'}, inplace=True)
    du_agg['DU_DATE'] -= pd.DateOffset(months=1)
    du_agg = du_agg[du_agg['DU_INDICATOR'] == True].drop_duplicates(['CUSTOMER'])
    
    ## list of customers who are drawing up
    du_customers = list(du_agg['CUSTOMER'])
    
    if len(du_customers) == 0:
        continue
    
    du_find = du_find[du_find['CUSTOMER'].isin(du_customers)].copy()
    
    du_find = du_find.groupby('CUSTOMER').apply(lambda group: group.iloc[:-1, 1:]).reset_index()
    du_find.drop('level_1', axis=1, inplace=True)
    
    du_find = du_find.merge(du_agg, left_on=['CUSTOMER'], right_on=['CUSTOMER'])
    
    du_find['DU_AVG_START'] = du_find['DU_DATE']  + pd.DateOffset(months=drawup_window)
    du_find['DU_AVG_END'] = du_find['DU_DATE']  + pd.DateOffset(months=drawup_window+statistics_period-1)
    
    du_find_12 = du_find[du_find['REVENUE_DATE'].between(du_find['DU_AVG_START'], du_find['DU_AVG_END'])].copy()
    
    du_stat = du_find_12.groupby(['CUSTOMER'], as_index=False).agg({'ACTIVE_CARD_COUNT':['mean','std']})
    
    du_stat.columns = ['CUSTOMER', 'mean_du','std_du']
    
    rise_df_ = du_agg.merge(du_stat, left_on='CUSTOMER', right_on='CUSTOMER', how='left')
    
    rise_df = pd.concat([rise_df, rise_df_], ignore_index=True)

rise_df.rename(columns={'DU_DATE':'DRAW_UP_DATE',
                        'mean_du':'MEAN_DU',
                       'std_du':'STD_DU'}, inplace=True)

rise_df = rise_df[['CUSTOMER','DRAW_UP_DATE','MEAN_DU','STD_DU']]
rise_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
CALCULATED_CARD_DRAW_UPS_df = rise_df

# Write recipe outputs
CALCULATED_CARD_DRAW_UPS = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS")
CALCULATED_CARD_DRAW_UPS.write_with_schema(CALCULATED_CARD_DRAW_UPS_df)
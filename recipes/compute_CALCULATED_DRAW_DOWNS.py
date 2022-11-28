# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from dateutil.relativedelta import relativedelta
from helper import *
import time

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

print("start_date", start_date)
print("end_date", end_date)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
NAFCUSTOMER_ACTIVE_CARDS_FULL = dataiku.Dataset("NAFCUSTOMER_ACTIVE_CARDS_FULL")
NAFCUSTOMER_ACTIVE_CARDS_FULL_df = NAFCUSTOMER_ACTIVE_CARDS_FULL.get_dataframe()
print(len(NAFCUSTOMER_ACTIVE_CARDS_FULL_df))
NAFCUSTOMER_ACTIVE_CARDS_FULL_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def date_tz_naive(pd_s):
    return pd.to_datetime(pd_s).apply(lambda x:x.tz_localize(None))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
customer_list_full = NAFCUSTOMER_ACTIVE_CARDS_FULL_df.CUSTOMER.unique()
print(len(customer_list_full))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def find_consistent_cust(df, consecutive=3):
    '''returns a list of customers who are consistent for 3 (default value) months'''

    ## Needs only these columns ['customer_account_name', 'revenue_month', 'purchase_gallons_qty']

    df = df[['CUSTOMER', 'REVENUE_DATE', 'ACTIVE_CARD_COUNT']].copy()
    df.sort_values(by=['CUSTOMER', 'REVENUE_DATE'], inplace=True)

    z = (df.groupby(['CUSTOMER'])['REVENUE_DATE'].diff(1)/np.timedelta64(1, 'M'))
    z = z.round(0)
    z = (z == 1).astype('int')
    df['CUST_CONS'] = (z * (z.groupby((z != z.shift()).cumsum()).cumcount() + 2))
    cust_cons = df.groupby('CUSTOMER')['CUST_CONS'].max()

    return list(cust_cons[cust_cons>=consecutive].index)

def add_padding_func(df, padding=12, last_date=None):
    '''
    Fills all the zeros in between for intermittent data and also fills the trailing data with
    12 zeros or till the last date whichever is earlier
    '''

    cols = ['CUSTOMER', 'REVENUE_DATE']

    common_cols = set(df.columns).intersection(set(cols))

    profile = df[common_cols].drop_duplicates()

    vol = df[['CUSTOMER', 'REVENUE_DATE', 'ACTIVE_CARD_COUNT']].copy()
    vol = vol.groupby(['CUSTOMER', 'REVENUE_DATE'])[['ACTIVE_CARD_COUNT']].sum().reset_index()
    vol.reset_index(drop=True, inplace=True)

    vol.sort_values(['CUSTOMER', 'REVENUE_DATE'], inplace=True)
    vol.reset_index(drop=True, inplace=True)

    last_rev_date = vol.groupby(['CUSTOMER'])[['REVENUE_DATE']].last()
    last_rev_date = last_rev_date[last_rev_date['REVENUE_DATE'] < pd.to_datetime(last_date)]
    last_rev_date['REVENUE_DATE'] = last_rev_date['REVENUE_DATE'] + pd.DateOffset(months=padding)
    last_rev_date['LAST_DATE'] = pd.to_datetime(last_date)
    last_rev_date['REVENUE_DATE'] = last_rev_date[['REVENUE_DATE','LAST_DATE']].min(axis=1)
    last_rev_date.drop(['LAST_DATE'], axis=1, inplace=True)
    last_rev_date.reset_index(inplace=True)
    vol = pd.concat([vol, last_rev_date], ignore_index=True)
    vol.fillna(0, inplace=True)
    vol = (vol.set_index('REVENUE_DATE').groupby('CUSTOMER').resample('MS').asfreq()
                  .drop(['CUSTOMER'], 1).reset_index())
    vol.fillna(0, inplace=True)
    df = vol.merge(profile, how='left', on = ['CUSTOMER', 'REVENUE_DATE'])
    df.fillna(method='ffill', inplace=True)

    return df

def find_average_func(dd_find, n=12):

    dd_find.sort_values(['CUSTOMER', 'REVENUE_DATE'], inplace=True)
    dd_find2 = dd_find.sort_values(['CUSTOMER', 'REVENUE_DATE'], ascending=[True, False]).reset_index(drop=True)

    dd_find.reset_index(drop=True, inplace=True)
    dd_find['LAST_N_MONTHS_AVG'] = dd_find.groupby(['CUSTOMER'])['ACTIVE_CARD_COUNT']\
                                        .rolling(n, min_periods=1).mean().reset_index(drop=True)
    dd_find2['NEXT_N_MONTHS_AVG'] = dd_find2.groupby(['CUSTOMER'])['ACTIVE_CARD_COUNT']\
                                        .rolling(n, min_periods=1).mean().reset_index(drop=True)

    dd_find['LAST_N_MONTHS_AVG'] = dd_find.groupby('CUSTOMER')['LAST_N_MONTHS_AVG'].shift(1)
    dd_find2['NEXT_N_MONTHS_AVG'] = dd_find2.groupby('CUSTOMER')['NEXT_N_MONTHS_AVG'].shift(1)

    dd_find = dd_find.merge(dd_find2[['CUSTOMER', 'REVENUE_DATE', 'NEXT_N_MONTHS_AVG']],
                on=['CUSTOMER', 'REVENUE_DATE'])

    return dd_find

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
page_size = 50000
idx = 0
current_page = 0
max_pages = 0

drop_df = pd.DataFrame()
t0 = time.time()

total_pages = len(customer_list_full)/page_size

while idx<len(customer_list_full):

    current_page+=1
    print("page", current_page)

    to_range = idx+page_size
    if to_range>len(customer_list_full):
        to_range = len(customer_list_full)-1

    current_set = customer_list_full[idx:to_range]

    #==============================================

    df_v = NAFCUSTOMER_ACTIVE_CARDS_FULL_df[NAFCUSTOMER_ACTIVE_CARDS_FULL_df.CUSTOMER.isin(current_set)]
    print("processing", len(df_v.CUSTOMER.unique()), "customers")
    print(len(df_v), "data frame records")

    df_v['REVENUE_DATE'] = df_v.REVENUE_MONTH.astype(str) + "/01/" + df_v.REVENUE_YEAR.astype(str)
    df_v['REVENUE_DATE'] = date_tz_naive(df_v['REVENUE_DATE'])

    df_v = df_v[['CUSTOMER','REVENUE_DATE', 'ACTIVE_CARD_COUNT']]

    df_v_max = df_v[['CUSTOMER','ACTIVE_CARD_COUNT']]
    df_max = df_v_max.groupby(by=["CUSTOMER"]).max().reset_index()
    df_max.columns = ['CUSTOMER', 'ACTIVE_CARD_MAX']

    df_v.dropna(subset=['CUSTOMER'], inplace=True)

    df_v['REVENUE_DATE'] = pd.to_datetime(df_v['REVENUE_DATE'])
    df_v.sort_values(['REVENUE_DATE'], inplace=True)

    seen_accounts = df_v[df_v['ACTIVE_CARD_COUNT'] > 0].groupby(['CUSTOMER'], as_index=False)[['REVENUE_DATE']].first()
    seen_accounts['FIRST_DATE'] = seen_accounts['REVENUE_DATE'] - pd.DateOffset(months=1)

    df = df_v
    period_end_date = end_date
    match_type = 'program_flip'
    period_start_date=None
    split=None

    drawdown = (100 - drawdown)/100
    drawdown_fwd_check /= 100

    inactive_date_start = pd.to_datetime(period_end_date) + relativedelta(months=-inactive_period)

    df = df[df['REVENUE_DATE'] <= period_end_date].copy()

    if period_start_date:
        period_start_date = pd.to_datetime(period_start_date)
        df = df[df['REVENUE_DATE'] >= period_start_date].copy()

    all_customer_names = list(df['CUSTOMER'].unique())

    dd_find = df[df['CUSTOMER'].isin(all_customer_names)].copy()

    consistent_customers_dd = find_consistent_cust(dd_find, consecutive=consistency)
    if len(consistent_customers_dd) == 0:
        continue

    dd_find = dd_find[dd_find['CUSTOMER'].isin(consistent_customers_dd)].copy()
    dd_find = add_padding_func(dd_find, padding=statistics_period, last_date=period_end_date)
    dd_find = find_average_func(dd_find, n=statistics_period)

    dd_find['DD_INDICATOR'] = np.where(((drawdown*(dd_find['LAST_N_MONTHS_AVG'].round(3)) >
                                     dd_find['ACTIVE_CARD_COUNT'].round(3)) &
                                    (dd_find['NEXT_N_MONTHS_AVG'].round(3) <
                                     drawdown_fwd_check*dd_find['LAST_N_MONTHS_AVG'].round(3))),
                                   True, False)

    ## Find the first drawdown and also the list of customers
    pflip_dd = dd_find[dd_find['DD_INDICATOR'] == True].copy()
    pflip_dd.drop_duplicates('CUSTOMER', inplace=True)
    first_drop_idx = pflip_dd.index
    pflip_dd_customers = list(dd_find['CUSTOMER'].unique())
    first_drop = dd_find.iloc[first_drop_idx]

    ## Identify the lookback period
    first_drop = first_drop[['CUSTOMER', 'REVENUE_DATE']].copy()
    first_drop = first_drop[first_drop['REVENUE_DATE'] <= inactive_date_start].copy()
    first_drop['START_DATE']  = first_drop['REVENUE_DATE'] - pd.DateOffset(months=drawdown_lookback_period)
    first_drop.rename(columns = {'REVENUE_DATE':'DD_DATE'}, inplace=True)
    dd_find_df = dd_find[dd_find['CUSTOMER'].isin(pflip_dd_customers)]
    dd_find_df = dd_find_df.merge(first_drop, on=['CUSTOMER'])
    dd_find_df = dd_find_df[dd_find_df['REVENUE_DATE'].between(dd_find_df['START_DATE'],dd_find_df['DD_DATE'])].copy()

    ## Compute the sharpest fall from the lookback period
    dd_find_df.sort_values(['CUSTOMER', 'REVENUE_DATE'], inplace=True)
    dd_find_df['DROP'] = dd_find_df.groupby(['CUSTOMER'])['ACTIVE_CARD_COUNT'].diff(-1)

    ## Find the corresponding period and remove duplicates in case of a similar values
    drop_idx = dd_find_df.groupby(['CUSTOMER'])['DROP'].transform(max) == dd_find_df['DROP']
    drop_month_df = dd_find_df[drop_idx].copy()
    drop_month_df.drop_duplicates(['CUSTOMER'], inplace=True)

    ## remove the first record
    dd_find = dd_find.groupby('CUSTOMER').apply(lambda group: group.iloc[1:, 1:]).reset_index()
    dd_find.drop('level_1', axis=1, inplace=True)

    ## Find the time periods for calculating statistics (mean and standard deviation)
    drop_month_df.rename(columns = {'REVENUE_DATE':'DROP_DATE'}, inplace=True)
    dd_find = dd_find.merge(drop_month_df[['CUSTOMER', 'DROP_DATE']], on='CUSTOMER')
    dd_find['END_DATE'] = dd_find['DROP_DATE'] - pd.DateOffset(months=3)
    dd_find['START_DATE'] = dd_find['END_DATE'] - pd.DateOffset(months=statistics_period-1)
    pflip_12_data = dd_find[dd_find['REVENUE_DATE'].between(dd_find['START_DATE'], dd_find['END_DATE'])].copy()

    ## Calculate Mean and Standard Deviation
    dd_stat = pflip_12_data.groupby(['CUSTOMER'], as_index=False).agg({'ACTIVE_CARD_COUNT':['mean','std']})
    dd_stat.columns = ['CUSTOMER_DD', 'MEAN_DD','STD_DD']
    drop_month_df = drop_month_df.merge(dd_stat,
                                        left_on='CUSTOMER',
                                        right_on='CUSTOMER_DD',
                                        how='left')

    drop_month_df = pd.merge(drop_month_df, df_max, on='CUSTOMER', how='left')
    drop_month_df = drop_month_df[['CUSTOMER','DROP_DATE','ACTIVE_CARD_MAX']]
    drop_month_df.columns = ['CUSTOMER','DRAW_DOWN_DATE','ACTIVE_CARD_MAX']

    print(len(drop_month_df), "new drop records")
    drop_df = pd.concat([drop_df, drop_month_df], ignore_index=True)

    print(len(drop_df), "total drop records")
    print("saving to snowflake...")
    CALCULATED_DRAW_DOWNS_df = drop_df
    CALCULATED_DRAW_DOWNS = dataiku.Dataset("CALCULATED_DRAW_DOWNS")
    CALCULATED_DRAW_DOWNS.write_with_schema(CALCULATED_DRAW_DOWNS_df)

    pages_remaining = total_pages-current_page

    t1 = time.time()
    avg_duration = (((t1-t0)/current_page)/60.0)
    print(round(avg_duration,2), "avg mins per iteration")
    print(round(pages_remaining,2), "pages remaining")
    print(round(avg_duration*pages_remaining,2), "estimated minutes remaining")
    print()

    #=====================================

    idx+=page_size

    if max_pages>0:
        if current_page>=max_pages:
            break;

print("done")
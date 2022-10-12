import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
import logging

helper_log = logging.getLogger('helper')
helper_log.setLevel(logging.INFO)

__all__ = ['find_consistent_customers', 'plot_trend', 'add_padding', 'find_average', 'split_list', 'preprocess_data']

def date_tz_naive(pd_s):
    return pd.to_datetime(pd_s).apply(lambda x:x.tz_localize(None))

def preprocess_data(df, start_date, end_date):
    df = df[df['revenue_date'].between(pd.to_datetime(start_date), pd.to_datetime(end_date))].copy()
    df = df.dropna(subset=['customer_account_id'])
    df = df[df['customer_source_system_code'].isin(['TANDEM', 'SIEBEL'])]
    df.rename(columns={'puchase_gallons_qty':'purchase_gallons_qty'}, inplace=True)

    ## data pre processing
    df['customer_account_id'] = df['customer_account_id'].astype('int64')
    df['revenue_date'] = pd.to_datetime(df['revenue_date'])

    states = list(df['account_state_prov_code'].unique())

    states_dict = {s:s.upper() for s in states}

    df['account_state_prov_code'] = df['account_state_prov_code'].map(states_dict)

    ## remove the unneccesary columns
    remove_cols=['revenue_month','revenue_year','gross_spend_amount']
    df = df.drop([x for x in remove_cols if x in df.columns], axis=1)

    ## group by all columns except purchase_gallons_qty
    groupby_cols = list(df.columns)
    groupby_cols.remove('purchase_gallons_qty')
    df.groupby(groupby_cols, as_index=False)['purchase_gallons_qty'].sum()
    df.sort_values(['revenue_date'], inplace=True)
    return df

def find_consistent_customers(df, consecutive=3):
    '''returns a list of customers who are consistent for 3 (default value) months'''
    
    ## Needs only these columns ['customer_account_name', 'revenue_month', 'purchase_gallons_qty']
    
    df = df[['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE', 'PURCHASE_GALLONS_QTY']].copy()
    df.sort_values(by=['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE'], inplace=True)
    
    z = (df.groupby(['CUSTOMER_ACCOUNT_ID'])['REVENUE_DATE'].diff(1)/np.timedelta64(1, 'M'))
    z = z.round(0)
    z = (z == 1).astype('int')
    df['CUST_CONS'] = (z * (z.groupby((z != z.shift()).cumsum()).cumcount() + 2))
    cust_cons = df.groupby('CUSTOMER_ACCOUNT_ID')['CUST_CONS'].max()
    
    return list(cust_cons[cust_cons>=consecutive].index)

def plot_trend(cust_id, df, trend=None):

    cols = ['customer_account_name', 'revenue_date', 'purchase_gallons_qty']
    dd_data = df[df['customer_account_id'] == cust_id][cols]
    dd_data.set_index('revenue_date', inplace=True)
    dd_data.sort_index(inplace=True)
    
    
    plt.figure(figsize=(10,6))
    sns.lineplot(data=dd_data, x='revenue_date', y='purchase_gallons_qty', 
                 marker='o'
                )#
    plt.ylim(0);

def add_padding(df, padding=12, last_date=None):
    '''
    Fills all the zeros in between for intermittent data and also fills the trailing data with 
    12 zeros or till the last date whichever is earlier
    '''
    cols = ['customer_account_id', 'customer_account_name', 'customer_name',
                           'account_since_date', 'customer_business_program_name', 'account_city',
                           'account_state_prov_code', 'customer_source_system_code', 'revenue_date']
    
    common_cols = set(df.columns).intersection(set(cols))
    
    profile = df[common_cols].drop_duplicates()

    vol = df[['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE', 'PURCHASE_GALLONS_QTY']].copy()
    vol = vol.groupby(['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE'])[['PURCHASE_GALLONS_QTY']].sum().reset_index()
    vol.reset_index(drop=True, inplace=True)

    vol.sort_values(['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE'], inplace=True)
    vol.reset_index(drop=True, inplace=True)

    last_rev_date = vol.groupby(['CUSTOMER_ACCOUNT_ID'])[['REVENUE_DATE']].last()
    last_rev_date = last_rev_date[last_rev_date['REVENUE_DATE'] < pd.to_datetime(last_date)]
    last_rev_date['REVENUE_DATE'] = last_rev_date['REVENUE_DATE'] + pd.DateOffset(months=padding)
    last_rev_date['LAST_DATE'] = pd.to_datetime(last_date)
    last_rev_date['REVENUE_DATE'] = last_rev_date[['REVENUE_DATE','LAST_DATE']].min(axis=1)
    last_rev_date.drop(['LAST_DATE'], axis=1, inplace=True)
    last_rev_date.reset_index(inplace=True)
    vol = pd.concat([vol, last_rev_date], ignore_index=True)
    vol.fillna(0, inplace=True)
    vol = (vol.set_index('REVENUE_DATE').groupby('CUSTOMER_ACCOUNT_ID').resample('MS').asfreq()
                  .drop(['CUSTOMER_ACCOUNT_ID'], 1).reset_index())
    vol.fillna(0, inplace=True)
    df = vol.merge(profile, how='left', on = ['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE'])
    df.fillna(method='ffill', inplace=True)

    return df

def find_average(dd_find, n=12):
   
    ## Old Method
    
    # dd_find.sort_values(['customer_account_id', 'revenue_date'], inplace=True)
    # dd_find.reset_index(drop=True, inplace=True)
    # dd_find['n_months_avg_start'] = dd_find['purchase_gallons_qty'].groupby(dd_find['customer_account_id'])\
    #                                     .rolling(n, min_periods=1).mean().reset_index(drop=True)
    # dd_find['n_months_avg_end'] = dd_find['purchase_gallons_qty'].groupby(dd_find['customer_account_id'])\
    #                                     .shift(-n).rolling(n, min_periods=1).mean().reset_index(drop=True)

    # dd_find['last_n_months_avg'] = dd_find.groupby('customer_account_id')['n_months_avg_start'].shift(1)
    # dd_find['next_n_months_avg'] = dd_find.groupby('customer_account_id')['n_months_avg_start'].shift(-n)
    # dd_find['next_n_months_avg'] = dd_find['next_n_months_avg'].fillna(dd_find['n_months_avg_end'])

    # dd_find.drop(['n_months_avg_start', 'n_months_avg_end'], axis=1, inplace=True)

    ## New Method

    dd_find.sort_values(['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE'], inplace=True)
    dd_find2 = dd_find.sort_values(['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE'], ascending=[True, False]).reset_index(drop=True)

    dd_find.reset_index(drop=True, inplace=True)
    dd_find['LAST_N_MONTHS_AVG'] = dd_find.groupby(['CUSTOMER_ACCOUNT_ID'])['PURCHASE_GALLONS_QTY']\
                                        .rolling(n, min_periods=1).mean().reset_index(drop=True)
    dd_find2['NEXT_N_MONTHS_AVG'] = dd_find2.groupby(['CUSTOMER_ACCOUNT_ID'])['PURCHASE_GALLONS_QTY']\
                                        .rolling(n, min_periods=1).mean().reset_index(drop=True)

    dd_find['LAST_N_MONTHS_AVG'] = dd_find.groupby('CUSTOMER_ACCOUNT_ID')['LAST_N_MONTHS_AVG'].shift(1)
    dd_find2['NEXT_N_MONTHS_AVG'] = dd_find2.groupby('CUSTOMER_ACCOUNT_ID')['NEXT_N_MONTHS_AVG'].shift(1)

    dd_find = dd_find.merge(dd_find2[['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE', 'NEXT_N_MONTHS_AVG']], 
                on=['CUSTOMER_ACCOUNT_ID', 'REVENUE_DATE'])

    return dd_find

def plot_matches(id_tdm,id_sbl, dataset_pd):
    
    tdm = dataset_pd[dataset_pd['customer_account_id'].isin(id_tdm)].copy()
    sbl = dataset_pd[dataset_pd['customer_account_id'].isin(id_sbl)].copy()
    
    tdm['customer_account_id'] = tdm['customer_account_id'].astype('str')
    sbl['customer_account_id'] = sbl['customer_account_id'].astype('str')
    
    tdm['Name'] = tdm['customer_account_id']+ ', ' + tdm['customer_account_name']
    sbl['Name'] = sbl['customer_account_id']+ ', ' + sbl['customer_account_name']
    
    tdm['revenue_date'] = pd.to_datetime(tdm['revenue_date'])
    sbl['revenue_date'] = pd.to_datetime(sbl['revenue_date'])


    combined_sales = pd.concat([tdm, sbl], ignore_index=True)
    combined_sales['customer_account_id'] = combined_sales['customer_account_id'].astype('str')
    agg_sales = combined_sales.groupby('revenue_date')['purchase_gallons_qty'].sum().reset_index()
    
   
    f, ax = plt.subplots(1, 1, figsize=(14,7))

    ## Individual Sales
    sns.lineplot(data=combined_sales, 
                 x='revenue_date', 
                 y='purchase_gallons_qty', 
                 hue='Name',
                 marker='o',
                 ax=ax
            )
    ax.set_ylim(bottom=-20)
    
    
def split_list(lst, n):
    '''
    Splits a list into almost equal n parts
    '''
    k, m = divmod(len(lst), n)
    return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
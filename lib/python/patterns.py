import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

## custom functions
from helper import *

def find_drawdowns(df, 
                   match_type, 
                   period_end_date,
                   period_start_date=None, 
                   consistency=4,
                   drawdown=90, 
                   drawdown_fwd_check=5,
                   drawdown_lookback_period=6,
                   statistics_period=12,
                   inactive_period=4,
                   split=None):
    '''
    Parameters
    ----------
        df : pandas.DataFrame
            Table containing monthly aggregated revenue and fuel purchased (in gallons)
        match_type : str
            Type of transfer 'conversion' or 'program_flip'
        period_end_date : str
            Date in 'YYYY-MM-DD' format to which data need to be considered
        period_start_date: str, None
            Date in 'YYYY-MM-DD' format from which data need to be considered
        consistency : int, 4
            Months a customer need to be active consistently to be considered consistent
        drawdown : int, 90
            Drop in BAU in percentage compared to average of last 'drawdown_period_average' months
        drawdown_fwd_check : int, 5
            Next 'drawdown_period_average' month average to stay below drawdown check %
        drawdown_lookback_period : int, 6
            Months to look back to calculate sharpest fall
        statistics_period : int, 12
            Period for which statistics need to be calculated
        split: int, None
            No of parts the customers need to be split to avoid MemoryError
    Returns
    -------
        drop_df : pandas.DataFrame
            Table containing list of drawdowns identified with month and magnitude of sharpest fall
    '''
    ## derived inputs
    drawdown = (100 - drawdown)/100
    drawdown_fwd_check /= 100

    inactive_date_start = pd.to_datetime(period_end_date) + relativedelta(months=-inactive_period)
    
    if match_type == 'conversion':
        df = df[df['customer_source_system_code'] == 'TANDEM'].copy()

    df = df[df['revenue_date'] <= period_end_date].copy()

    if period_start_date:
        period_start_date = pd.to_datetime(period_start_date)
        df = df[df['revenue_date'] >= period_start_date].copy()

    ## list of all the accounts
    all_account_ids = list(df['customer_account_id'].unique())

    if not split:
        split=1

    all_account_ids_n = list(split_list(all_account_ids, split))

    drop_df = pd.DataFrame()

    for sublist in tqdm(all_account_ids_n):
        dd_find = df[df['customer_account_id'].isin(sublist)].copy()
        
        ## Find consistent customers
        consistent_customers_dd = find_consistent_customers(dd_find, consecutive=consistency)
        if len(consistent_customers_dd) == 0:
            continue

        dd_find = dd_find[dd_find['customer_account_id'].isin(consistent_customers_dd)].copy()

        ## Add padding, find the n months average and compute drawdown indicator based on the rules
        dd_find = add_padding(dd_find, padding=statistics_period, last_date=period_end_date)
        dd_find = find_average(dd_find, n=statistics_period)

        dd_find['dd_indicator'] = np.where(((drawdown*(dd_find['last_n_months_avg'].round(3)) > 
                                             dd_find['purchase_gallons_qty'].round(3)) &
                                            (dd_find['next_n_months_avg'].round(3) < 
                                             drawdown_fwd_check*dd_find['last_n_months_avg'].round(3))),
                                           True, False)
        ## Find the first drawdown and also the list of customers
        pflip_dd = dd_find[dd_find['dd_indicator'] == True].copy()
        pflip_dd.drop_duplicates('customer_account_id', inplace=True)
        first_drop_idx = pflip_dd.index
        pflip_dd_customers = list(dd_find['customer_account_id'].unique())
        first_drop = dd_find.iloc[first_drop_idx]

        ## Identify the lookback period
        first_drop = first_drop[['customer_account_id', 'revenue_date']].copy()
        first_drop = first_drop[first_drop['revenue_date'] <= inactive_date_start].copy()
        first_drop['start_date']  = first_drop['revenue_date'] - pd.DateOffset(months=drawdown_lookback_period)
        first_drop.rename(columns = {'revenue_date':'dd_date'}, inplace=True)
        dd_find_df = dd_find[dd_find['customer_account_id'].isin(pflip_dd_customers)]
        dd_find_df = dd_find_df.merge(first_drop, on=['customer_account_id'])
        dd_find_df = dd_find_df[dd_find_df['revenue_date'].between(dd_find_df['start_date'], 
                                                                   dd_find_df['dd_date'])].copy()

        ## Compute the sharpest fall from the lookback period
        dd_find_df.sort_values(['customer_account_id', 'revenue_date'], inplace=True)
        dd_find_df['drop'] = dd_find_df.groupby(['customer_account_id'])['purchase_gallons_qty'].diff(-1)

        ## Find the corresponding period and remove duplicates in case of a similar values
        drop_idx = dd_find_df.groupby(['customer_account_id'])['drop'].transform(max) == dd_find_df['drop']
        drop_month_df = dd_find_df[drop_idx].copy()
        drop_month_df.drop_duplicates(['customer_account_id'], inplace=True)
        
        ## remove the first record
        dd_find = dd_find.groupby('customer_account_id').apply(lambda group: group.iloc[1:, 1:]).reset_index()
        dd_find.drop('level_1', axis=1, inplace=True)

        ## Find the time periods for calculating statistics (mean and standard deviation)
        drop_month_df.rename(columns = {'revenue_date':'drop_date'}, inplace=True)
        dd_find = dd_find.merge(drop_month_df[['customer_account_id', 'drop_date']], on='customer_account_id')
        dd_find['end_date'] = dd_find['drop_date'] - pd.DateOffset(months=3)
        dd_find['start_date'] = dd_find['end_date'] - pd.DateOffset(months=statistics_period-1)
        pflip_12_data = dd_find[dd_find['revenue_date'].between(dd_find['start_date'], dd_find['end_date'])].copy()

        ## Calculate Mean and Standard Deviation
        dd_stat = pflip_12_data.groupby(['customer_account_id'], as_index=False).agg({'purchase_gallons_qty':['mean','std']})
        dd_stat.columns = ['customer_account_id_dd', 'mean_dd','std_dd']
        drop_month_df = drop_month_df.merge(dd_stat, 
                                            left_on='customer_account_id', 
                                            right_on='customer_account_id_dd',
                                            how='left')

        drop_df = pd.concat([drop_df, drop_month_df], ignore_index=True)
    
    drop_df.drop(['customer_account_id_dd'], axis=1, inplace=True)
    drop_df.rename(columns={'customer_account_id':'customer_account_id_dd', 
                            'customer_account_name': 'customer_account_name_dd',
                            'customer_name': 'customer_name_dd'}, inplace=True)

    return drop_df[['customer_account_id_dd', 'customer_account_name_dd', 'customer_name_dd', 
                    'dd_date', 'mean_dd', 'std_dd', 'customer_source_system_code', 'account_city',
                    'account_state_prov_code', 'customer_business_program_name']]

def find_drawups(df, 
                 match_type, 
                 period_start_date,
                 period_end_date=None, 
                 drawup_window=3,
                 statistics_period=12, 
                 split=None):
    '''
    Parameters
    ----------
        df : pandas.DataFrame
            Table containing monthly aggregated revenue and fuel purchased (in gallons)
        match_type : str
            Type of transfer 'conversion' or 'program_flip'
        period_start_date: str
            Date in 'YYYY-MM-DD' format from which data need to be considered
        period_end_date : str, None
            Date in 'YYYY-MM-DD' format to which data need to be considered
        drawup_window : int, 3
            Estimated number of months for complete drawup
        statistics_period : int, 12
            Period for which statistics need to be calculated
        split: int, None
            No of parts the customers need to be split to avoid MemoryError
    Returns
    -------
        rise_df : pandas.DataFrame
            Table containing list of drawups identified with month and magnitude of sharpest fall
    '''

    if match_type == 'conversion':
        df = df[df['customer_source_system_code'] == 'SIEBEL'].copy()

    period_start_date = pd.to_datetime(period_start_date)
    df = df[df['revenue_date'] >= period_start_date].copy()

    if period_end_date:
        period_end_date = pd.to_datetime(period_end_date)
        df = df[df['revenue_date'] <= period_end_date].copy()

    ## list of all the accounts
    all_account_ids = list(df['customer_account_id'].unique())

    if not split:
        split=1

    all_account_ids_n = list(split_list(all_account_ids, split))

    rise_df = pd.DataFrame()

    for sublist in tqdm(all_account_ids_n):

        du_find = df[df['customer_account_id'].isin(sublist)].copy()

        ## Filter Non-Zero Records and find the first non zero transaction date
        du_find = du_find[du_find['purchase_gallons_qty'] > 0]
        
        du_find.sort_values(['revenue_date'], inplace=True)
        du_agg = du_find.groupby(['customer_account_id',
                              'customer_account_name',
                              'customer_name',
                              'customer_source_system_code',
                              'account_city',
                              'account_state_prov_code',
                              'customer_business_program_name'], as_index=False)[['revenue_date']].min()

        du_agg['du_indicator'] = np.where((du_agg['revenue_date'] > period_start_date), True, False)
        du_agg.rename(columns={'revenue_date':'du_date'}, inplace=True)
        du_agg['du_date'] -= pd.DateOffset(months=1)
        du_agg = du_agg[du_agg['du_indicator'] == True].drop_duplicates(['customer_account_id'])

        ## list of customers who are drawing up
        du_customers = list(du_agg['customer_account_id'])
        
        if len(du_customers) == 0:
            continue
            
        du_find = du_find[du_find['customer_account_id'].isin(du_customers)].copy()
        
        ## remove the last record
        du_find = du_find.groupby('customer_account_id').apply(lambda group: group.iloc[:-1, 1:]).reset_index()
        du_find.drop('level_1', axis=1, inplace=True)
        
        du_find = du_find.merge(du_agg, left_on=['customer_account_id'], right_on=['customer_account_id'])
        
        du_find['du_avg_start'] = du_find['du_date']  + pd.DateOffset(months=drawup_window)
        du_find['du_avg_end'] = du_find['du_date']  + pd.DateOffset(months=drawup_window+statistics_period-1)

        du_find_12 = du_find[du_find['revenue_date'].between(du_find['du_avg_start'],
                                                          du_find['du_avg_end'])].copy()

        du_stat = du_find_12.groupby(['customer_account_id'], as_index=False).agg({'purchase_gallons_qty':['mean','std']})

        du_stat.columns = ['customer_account_id', 'mean_du','std_du']

        rise_df_ = du_agg.merge(du_stat, 
                left_on='customer_account_id', 
                right_on='customer_account_id',
                how='left')

        rise_df = pd.concat([rise_df, rise_df_], ignore_index=True)
    
    rise_df.rename(columns={'customer_account_id':'customer_account_id_du',
                        'customer_account_name': 'customer_account_name_du',
                        'customer_name': 'customer_name_du'}, inplace=True)

    return rise_df[['customer_account_id_du', 'customer_account_name_du', 'customer_name_du',
                    'du_date','customer_source_system_code', 'account_city', 
                    'account_state_prov_code', 'customer_business_program_name','mean_du','std_du']]
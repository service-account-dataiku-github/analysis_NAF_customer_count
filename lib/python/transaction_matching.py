import pandas as pd
import numpy as np
import Levenshtein
from tqdm.contrib.itertools import product as product_tqdm
from itertools import product
from joblib import Parallel, delayed
city_distance_dict={}
import multiprocessing as mp
cpu_count = int(0.75*mp.cpu_count())

def city_distance(city1, city2):
    global city_distance_dict
    if city1 == city2:
        city_distance_dict[(city1, city2)] = 0
        return 0
    if (city1, city2) in city_distance_dict:
        return city_distance_dict[(city1, city2)]
    elif (city2, city1) in city_distance_dict:
        return city_distance_dict[(city2, city1)]
    else:
        city_distance_dict[(city1, city2)] = Levenshtein.distance(city1, city2)
        return city_distance_dict[(city1, city2)]


def transaction_match(dd_df, du_df, 
                        match_type=None,
                        month_diff_h=6,
                        month_diff_l=0,
                        sd_mul=2,
                        max_city_distance=3):

    if match_type == 'conversion':
        dd_df = dd_df[dd_df['customer_source_system_code'] == 'TANDEM'].copy()
        du_df = du_df[du_df['customer_source_system_code'] == 'SIEBEL'].copy()
    
    txn_matches = pd.DataFrame()
    
    states = list(dd_df['account_state_prov_code'].unique())
    dates = sorted(list(dd_df['dd_date'].unique()))
    
    dd_source = ['TANDEM']
    du_source = ['SIEBEL']
    if match_type == 'program_flip':
        dd_source = ['TANDEM', 'SIEBEL']
        du_source = ['TANDEM', 'SIEBEL']
    
    for state, date, dd_s, du_s in product_tqdm(states, dates, dd_source, du_source):
        ## Filter the specific month and TANDEM
        if match_type == 'program_flip' and dd_s != du_s:
            continue
        dd_month = dd_df[(dd_df['dd_date'] == pd.to_datetime(date)) & 
                         (dd_df['account_state_prov_code'] == state) & 
                         (dd_df['customer_source_system_code'] == dd_s)].copy()
        du_window = du_df[(du_df['du_date'] >= (pd.to_datetime(date) - pd.DateOffset(months=month_diff_h))) & 
                            (du_df['du_date'] < pd.to_datetime(date)) & 
                            (du_df['account_state_prov_code'] == state) & 
                            (du_df['customer_source_system_code'] == du_s)].copy() ## 1-6
        if dd_month.shape[0] == 0 or du_window.shape[0] == 0:
            continue
#        if match_type=='program_flip':
#            conditions_tuple = zip(du_window['du_date'].dt.to_period('M').astype(int),
#                                du_window['mean_du'],
#                                du_window['customer_business_program_name'])
#            conditions = np.array([((dd_month['dd_date'].dt.to_period('M').astype(int) - du_date) <= month_diff_h) &
#                                    ((dd_month['dd_date'].dt.to_period('M').astype(int) - du_date) > month_diff_l) &
#                                    ((dd_month['mean_dd'] - sd_mul*dd_month['std_dd']) <= mean) & 
#                                    (mean <= (dd_month['mean_dd'] + sd_mul*dd_month['std_dd'])) &
#                                    ((np.abs(mean - dd_month['mean_dd'])/dd_month['mean_dd']) < 0.5) &
#                                    (dd_month['customer_business_program_name'] != business_prog)
#                                        for du_date, mean, business_prog in conditions_tuple])
#        elif match_type=='conversion':
        conditions_tuple = zip(du_window['du_date'].dt.to_period('M').astype(int),
                                du_window['mean_du'])
        conditions = np.array([((dd_month['dd_date'].dt.to_period('M').astype(int) - du_date) <= month_diff_h) &
                                ((dd_month['dd_date'].dt.to_period('M').astype(int) - du_date) > month_diff_l) &
                                ((dd_month['mean_dd'] - sd_mul*dd_month['std_dd']) <= mean) & 
                                (mean <= (dd_month['mean_dd'] + sd_mul*dd_month['std_dd'])) & 
                                ((np.abs(mean - dd_month['mean_dd'])/dd_month['mean_dd']) < 0.5)
                                    for du_date, mean in conditions_tuple])
#        else:
#            raise ValueError('Unknown match type')
        
        conditions_t = list(conditions.T)
        ids_flat = du_window['customer_account_id_du'].values
        ids = list(np.tile(ids_flat, (len(dd_month), 1)))

        dd_month['customer_account_id_du'] = [ids_[i] for ids_,i in zip(ids, conditions_t)]
        dd_month = dd_month.explode('customer_account_id_du')
        dd_month_merged = dd_month.merge(du_window, how='left', 
                                        left_on='customer_account_id_du', 
                                        right_on='customer_account_id_du',
                                        suffixes=('_dd', '_du'))
        dd_month_merged.dropna(axis=0, subset=['customer_account_id_du'], inplace=True)

        if dd_month_merged.shape[0] == 0:
            continue
        dd_month_merged['city_distance'] = (dd_month_merged.loc[:, ['account_city_dd', 'account_city_du']]
                                            .apply(lambda x: city_distance(*x), axis=1))
        dd_month_merged = dd_month_merged[dd_month_merged['city_distance'] <= max_city_distance].copy()
        
        txn_matches = pd.concat([txn_matches, dd_month_merged], ignore_index=True)

    return txn_matches

def transaction_match_mp(dd_df, du_df, 
                        match_type=None,
                        month_diff_h=6,
                        month_diff_l=0,
                        sd_mul=2,
                        max_city_distance=3):

    if match_type == 'conversion':
        dd_df = dd_df[dd_df['customer_source_system_code'] == 'TANDEM'].copy()
        du_df = du_df[du_df['customer_source_system_code'] == 'SIEBEL'].copy()
    
    txn_matches = pd.DataFrame()
    
    states = list(dd_df['account_state_prov_code'].unique())
    dates = sorted(list(dd_df['dd_date'].unique()))
    
    dd_source = ['TANDEM']
    du_source = ['SIEBEL']
    if match_type == 'program_flip':
        dd_source = ['TANDEM', 'SIEBEL']
        du_source = ['TANDEM', 'SIEBEL']
    
    results = Parallel(n_jobs=cpu_count)(delayed(transaction_match_mp_i)(dd_df, du_df, 
                                                            match_type,
                                                           state, date, 
                                                           dd_s, du_s, 
                                                           month_diff_h, month_diff_l,
                                                           sd_mul, max_city_distance) for state, date, dd_s, du_s in \
                             product_tqdm(states, dates, dd_source, du_source))
    txn_matches = pd.concat(results).reset_index(drop=True)                      

    return txn_matches

def transaction_match_mp_i(dd_df, du_df, 
                        match_type,
                        state, date, 
                        dd_s, du_s, 
                        month_diff_h, month_diff_l,
                        sd_mul, max_city_distance):
    if match_type == 'program_flip' and dd_s != du_s:
        return pd.DataFrame()
    dd_month = dd_df[(dd_df['dd_date'] == pd.to_datetime(date)) & 
                     (dd_df['account_state_prov_code'] == state) & 
                     (dd_df['customer_source_system_code'] == dd_s)].copy()
    du_window = du_df[(du_df['du_date'] >= (pd.to_datetime(date) - pd.DateOffset(months=month_diff_h))) & 
                        (du_df['du_date'] < pd.to_datetime(date)) & 
                        (du_df['account_state_prov_code'] == state) & 
                        (du_df['customer_source_system_code'] == du_s)].copy() ## 1-6
    if dd_month.shape[0] == 0 or du_window.shape[0] == 0:
        return pd.DataFrame()
    if match_type=='program_flip':
        conditions_tuple = zip(du_window['du_date'].dt.to_period('M').astype(int),
                            du_window['mean_du']
                            #, du_window['customer_business_program_name']
                              )
        conditions = np.array([((dd_month['dd_date'].dt.to_period('M').astype(int) - du_date) <= month_diff_h) &
                                ((dd_month['dd_date'].dt.to_period('M').astype(int) - du_date) > month_diff_l) &
                                ((dd_month['mean_dd'] - sd_mul*dd_month['std_dd']) <= mean) & 
                                (mean <= (dd_month['mean_dd'] + sd_mul*dd_month['std_dd'])) &
                                ((np.abs(mean - dd_month['mean_dd'])/dd_month['mean_dd']) < 0.5)
                               # (dd_month['customer_business_program_name'] != business_prog)
                                    for du_date, mean
                               # , business_prog 
                               in conditions_tuple])
    elif match_type=='conversion':
        conditions_tuple = zip(du_window['du_date'].dt.to_period('M').astype(int),
                                du_window['mean_du'])
        conditions = np.array([((dd_month['dd_date'].dt.to_period('M').astype(int) - du_date) <= month_diff_h) &
                                ((dd_month['dd_date'].dt.to_period('M').astype(int) - du_date) > month_diff_l) &
                                ((dd_month['mean_dd'] - sd_mul*dd_month['std_dd']) <= mean) & 
                                (mean <= (dd_month['mean_dd'] + sd_mul*dd_month['std_dd'])) & 
                                ((np.abs(mean - dd_month['mean_dd'])/dd_month['mean_dd']) < 0.5)
                                    for du_date, mean in conditions_tuple])
    else:
        raise ValueError('Unknown match type')

    conditions_t = list(conditions.T)
    ids_flat = du_window['customer_account_id_du'].values
    ids = list(np.tile(ids_flat, (len(dd_month), 1)))

    dd_month['customer_account_id_du'] = [ids_[i] for ids_,i in zip(ids, conditions_t)]
    dd_month = dd_month.explode('customer_account_id_du')
    dd_month_merged = dd_month.merge(du_window, how='left', 
                                    left_on='customer_account_id_du', 
                                    right_on='customer_account_id_du',
                                    suffixes=('_dd', '_du'))
    dd_month_merged.dropna(axis=0, subset=['customer_account_id_du'], inplace=True)

    if dd_month_merged.shape[0] == 0:
        return pd.DataFrame()
    dd_month_merged['city_distance'] = (dd_month_merged.loc[:, ['account_city_dd', 'account_city_du']]
                                        .apply(lambda x: city_distance(*x), axis=1))
    dd_month_merged = dd_month_merged[dd_month_merged['city_distance'] <= max_city_distance].copy()
    return dd_month_merged
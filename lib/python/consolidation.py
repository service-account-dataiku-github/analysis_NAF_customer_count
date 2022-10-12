from collections import defaultdict
import os
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from helper import split_list, find_consistent_customers

__all__ = ['combine_matches', 'consolidate_matches', 'find_attritions', 
'find_new_accounts', 'get_attrition_status', 'get_new_account_status']

def fuzzy_scorer(combined_C2):
    scores = defaultdict(list)
    
    for i, r in combined_C2.iterrows():        
        acc_dd = r['customer_account_name_dd']
        acc_du = r['customer_account_name_du']
        
        cus_dd = r['customer_name_dd']
        cus_du = r['customer_name_du']
        
        prog_dd = r['business_program_dd']
        prog_du = r['business_program_du']
        
        #SCORER
        if acc_dd == 'nan' and acc_du == 'nan':
            scores['acc_dd_acc_du'].append(0)
        else:
            scores['acc_dd_acc_du'].append(fuzz.token_sort_ratio(acc_dd, acc_du))
        
        if cus_dd == 'nan' and cus_du == 'nan':
            scores['cus_dd_cus_du'].append(0)
        else:
            scores['cus_dd_cus_du'].append(fuzz.token_sort_ratio(cus_dd, cus_du))
        
        if acc_dd == 'nan' and cus_du == 'nan':
            scores['acc_dd_cus_du'].append(0)
        else:
            scores['acc_dd_cus_du'].append(fuzz.token_sort_ratio(acc_dd, cus_du))
        
        if acc_du == 'nan' and cus_dd == 'nan':
            scores['cus_dd_acc_du'].append(0)
        else:
            scores['cus_dd_acc_du'].append(fuzz.token_sort_ratio(acc_du, cus_dd))
        
        if cus_dd == 'nan' and prog_dd == 'nan':
            scores['cus_dd_prog_dd'].append(0)
        else:
            scores['cus_dd_prog_dd'].append(fuzz.token_sort_ratio(cus_dd, prog_dd))
        
        if cus_du == 'nan' and prog_du == 'nan':
            scores['cus_du_prog_du'].append(0)
        else:
            scores['cus_du_prog_du'].append(fuzz.token_sort_ratio(cus_du, prog_du))
        
    return scores
    

def combine_matches(txn_matches, name_matches, dd_df, du_df, sd_multiplier=2):
    txn_matches = txn_matches[['customer_account_id_dd', 'customer_account_id_du',
                           'customer_account_name_dd', 'customer_account_name_du']].copy()

    name_matches = name_matches[['customer_account_id_dd', 'customer_account_id_du',
                            'customer_account_name_dd', 'customer_account_name_du',
                            'name_score']].copy()

    combined_matches = name_matches.merge(txn_matches,
                    on=['customer_account_id_dd', 'customer_account_id_du'],
                    how='outer')

    combined_matches['customer_account_name_du'] = combined_matches['customer_account_name_du_x'].combine_first(combined_matches['customer_account_name_du_y'])
    combined_matches['customer_account_name_dd'] = combined_matches['customer_account_name_dd_x'].combine_first(combined_matches['customer_account_name_dd_y'])

    combined_matches = combined_matches.merge(du_df[['customer_account_id_du', 'du_date', 'mean_du',
                            'customer_source_system_code']],
                        how='left', on='customer_account_id_du')\
                    .merge(dd_df[['customer_account_id_dd', 'dd_date', 'mean_dd', 'std_dd',
                            'customer_source_system_code']],
                        how='left', on='customer_account_id_dd', suffixes=('_du','_dd'))


    combined_matches['txn_score'] = np.abs(combined_matches['mean_dd'] - combined_matches['mean_du'])*100/combined_matches['mean_du']
    combined_matches.loc[combined_matches['customer_account_name_dd_y'].isna(), 'txn_score'] = np.nan
    combined_matches.drop(['customer_account_name_dd_x', 'customer_account_name_dd_y',
            'customer_account_name_du_x', 'customer_account_name_du_y'],
            axis=1, inplace=True)

    combined_matches['name_score'].fillna(0, inplace=True)
    combined_matches['txn_score'].fillna(500, inplace=True)

    combined_matches['n12_upper'] = combined_matches['mean_dd'] + sd_multiplier*combined_matches['std_dd']
    combined_matches['n12_lower'] = combined_matches['mean_dd'] - sd_multiplier*combined_matches['std_dd']

    segment_mapping = (('C1', (combined_matches['name_score'] > 70) & (combined_matches['txn_score'] < 50)),
            ('C2', (combined_matches['name_score'] <= 70) & (combined_matches['txn_score'] < 20)),
            ('C3', (combined_matches['name_score'] >= 80 ) & (combined_matches['mean_du'] < combined_matches['n12_lower'])),
            ('C4', (combined_matches['name_score'] >= 80) & (combined_matches['mean_du'] > combined_matches['n12_upper'])),
            ('C5', (combined_matches['name_score'] >= 80)))

    choices, conditions = zip(*segment_mapping)
    choices = list(choices)
    conditions = list(conditions)

    combined_matches['segment'] = np.select(conditions,
                        choices,
                        default='Cn')
    return combined_matches

def consolidate_matches(convs, flips, raw_df):
    convs = convs.copy()
    flips = flips.copy()
    
    convs['match_type'] = 'conversion'
    flips['match_type'] = np.where(flips['business_program_dd'] == flips['business_program_du'], 'migration', 'program flip')

    convs = convs[['customer_account_id_dd', 'customer_account_id_du',
       'customer_account_name_dd', 'customer_account_name_du', 'customer_name_dd', 'customer_name_du',
       'name_score', 'txn_score', 'dd_date', 'du_date', 'mean_du',
       'n12_upper', 'n12_lower', 'segment', 'customer_source_system_code_dd', 'customer_source_system_code_du', 
       'business_program_dd', 'business_program_du', 'match_type']]

    flips = flips[['customer_account_id_dd', 'customer_account_id_du',
        'customer_account_name_dd', 'customer_account_name_du', 'customer_name_dd', 'customer_name_du',
        'name_score', 'txn_score', 'dd_date', 'du_date', 'mean_du',
        'n12_upper', 'n12_lower', 'segment', 'customer_source_system_code_dd', 'customer_source_system_code_du', 
        'business_program_dd', 'business_program_du', 'match_type']]

    combined = flips.append(convs)
    combined_wd = combined.copy()

    combined_C2 = combined[combined['segment'] == 'C2'].copy()
    combined_not_C2 = combined[combined['segment'].isin(['C1', 'C3', 'C4', 'C5'])].copy()

    ## Threshold = 0.65 on max score
    ## Creating a new class for C2x based on the threshold


    #convert names to str, to handle NaNs during token sort ratio
    combined_C2['customer_account_name_dd'] = combined_C2['customer_account_name_dd'].astype('str')
    combined_C2['customer_account_name_du'] = combined_C2['customer_account_name_du'].astype('str')
    combined_C2['customer_name_dd'] = combined_C2['customer_name_dd'].astype('str')
    combined_C2['customer_name_du'] = combined_C2['customer_name_du'].astype('str')
    combined_C2['business_program_dd'] = combined_C2['business_program_dd'].astype('str')
    combined_C2['business_program_du'] = combined_C2['business_program_du'].astype('str')

    scores = fuzzy_scorer(combined_C2)

    combined_C2['acc_dd_acc_du'] = scores['acc_dd_acc_du']
    combined_C2['cus_dd_cus_du'] = scores['cus_dd_cus_du']
    combined_C2['acc_dd_cus_du'] = scores['acc_dd_cus_du']
    combined_C2['cus_dd_acc_du'] = scores['cus_dd_acc_du']
    combined_C2['cus_dd_prog_dd'] = scores['cus_dd_prog_dd']
    combined_C2['cus_du_prog_du'] = scores['cus_du_prog_du']

    combined_C2['secondary_name_score'] = np.where((combined_C2.cus_dd_prog_dd >= 70) | \
                                              (combined_C2.cus_du_prog_du >= 70) | \
                                              (combined_C2.customer_name_dd.str.contains('Universal Primary')) | \
                                              (combined_C2.customer_name_du.str.contains('Universal Primary')),
                                               combined_C2[['acc_dd_cus_du',
                                                           'cus_dd_acc_du',
                                                           'acc_dd_acc_du']].max(axis=1),
                                               combined_C2[['cus_dd_cus_du', 
                                                           'acc_dd_cus_du',
                                                           'cus_dd_acc_du',
                                                           'acc_dd_acc_du']].max(axis=1))

    combined_C2.loc[combined_C2['secondary_name_score'] < 65, 'segment'] = 'C2x'

    combined = combined_C2.append(combined_not_C2)
    c1_matches = combined[combined['segment'] == 'C1'].copy()

    c1_matches.sort_values(['name_score', 'txn_score'], ascending=[False, True], inplace=True)
    c1_matches.drop_duplicates(['customer_account_id_dd'], inplace=True)
    c1_matches.drop_duplicates(['customer_account_id_du'], inplace=True)

    c1_dd = list(c1_matches['customer_account_id_dd'].unique())
    c1_du = list(c1_matches['customer_account_id_du'].unique())

    cx_matches = combined[(combined['segment'].isin(['C3','C4','C5'])) & 
                                (~combined['customer_account_id_dd'].isin(c1_dd)) &
                                (~combined['customer_account_id_du'].isin(c1_du))
                                ].copy()

    cx_matches.sort_values(['name_score', 'txn_score'], ascending=[False, True], inplace=True)
    cx_matches.drop_duplicates(['customer_account_id_dd'], inplace=True)
    cx_matches.drop_duplicates(['customer_account_id_du'], inplace=True)

    cx_dd = list(cx_matches['customer_account_id_dd'].unique())
    cx_du = list(cx_matches['customer_account_id_du'].unique())

    c2_matches = combined[(combined['segment'].isin(['C2','C2x'])) & 
                                (~combined['customer_account_id_dd'].isin(c1_dd + cx_dd)) &
                                (~combined['customer_account_id_du'].isin(c1_du + cx_du))
                                ].copy()

    c2_matches.sort_values(['secondary_name_score', 'txn_score'], ascending=[False, True], inplace=True)
    c2_matches.drop_duplicates(['customer_account_id_dd'], inplace=True)
    c2_matches.drop_duplicates(['customer_account_id_du'], inplace=True)

    combined_matches = pd.concat([c1_matches, c2_matches, cx_matches], ignore_index=True)

    segment_mapping = {'C1': 'perfect match',
                   'C2': 'confirmed match, low name score with high txn score',
                   'C3': 'match + attrition',
                   'C4': 'match + consolidation/addition',
                   'C5': 'match with low txn score',
                   'C2x': 'unlikely match, low name score with high txn score, for further review'}

    combined_matches['class_code'] = combined_matches['segment'].map(segment_mapping)

    c4 = combined_matches[(combined_matches['segment'] == 'C4')].sort_values(['mean_du'], ascending=False)
    matched_dd_ids = list(combined_matches[combined_matches['segment'] != 'C2x']['customer_account_id_dd'].unique())
    matched_du_ids = list(combined_matches[combined_matches['segment'] != 'C2x']['customer_account_id_du'].unique())
    probable_mergers = list(zip(c4['customer_account_id_dd'], c4['customer_account_id_du'], c4['du_date'], c4['mean_du']))

    threshold = 70
    average_months = 12
    sd_multiplier = 3

    mergers_df = pd.DataFrame()

    for dd_id, du_id, rise_date, mean_du in tqdm(probable_mergers):
    #     ids = [i for i in matched_dd_ids if i != dd_id]
        merger = combined_wd[(combined_wd['customer_account_id_du'] == du_id) & 
                            (combined_wd['name_score'] >= threshold)].copy() 
        
        if merger.shape[0] <= 1:
            continue

        conditions = [merger['customer_account_id_dd'] == dd_id, 
                    merger['customer_account_id_dd'].isin(matched_dd_ids),
                    merger['customer_account_id_dd'].isin(matched_du_ids)]
        
        choices =    ['original', 
                    'has another match as a Drawdown', 
                    'has another match as a Drawup']
        
        merger['match type'] = np.select(conditions, choices, default='merger')

        ids_ham_as_du = list(merger[merger['match type'] == 'has another match as a Drawup']['customer_account_id_dd'])
        ids_to_be_eliminated = list(combined_matches[(combined_matches['segment'] != 'C2x') 
                                    & (combined_matches['customer_account_id_du'].isin(ids_ham_as_du))]['customer_account_id_dd'])
        merger = merger[~merger['customer_account_id_dd'].isin(ids_to_be_eliminated)]
        
        
        end_date = pd.to_datetime(rise_date)
        start_date = end_date - pd.DateOffset(months = (average_months-1))
        dd_ids = list(merger[~(merger['match type'].isin(['has another match as a Drawdown','has another match as a Drawup']))]['customer_account_id_dd'])

        raw_df_dds = raw_df[(raw_df['customer_account_id'].isin(dd_ids)) & 
                            (raw_df['revenue_date'] >= start_date) & 
                            (raw_df['revenue_date'] <= end_date)]

        merged_ts = raw_df_dds.groupby(['revenue_date'])['purchase_gallons_qty'].sum().reset_index()

        mean_dds = merged_ts['purchase_gallons_qty'].mean()
        std_dds = merged_ts['purchase_gallons_qty'].std()
        lb, ub = mean_dds - sd_multiplier*std_dds, mean_dds + sd_multiplier*std_dds

        if lb <= mean_du <= ub:
            mergers_df = pd.concat([mergers_df, merger])
    if not mergers_df.empty:
        mergers_df = mergers_df[mergers_df['match type'].isin(['original', 'merger'])].copy()
        mergers_df.sort_values(['match type'], ascending=False, inplace=True)
        mergers_df.drop_duplicates('customer_account_id_dd', inplace=True)

        dds_count = mergers_df.groupby(['customer_account_id_du']).size()
        multiple_match = list(dds_count[dds_count > 1].index)

        combined_matches = combined_matches[~combined_matches['customer_account_id_du'].isin(multiple_match)].copy()

        mergers_df = mergers_df[mergers_df['customer_account_id_du'].isin(multiple_match)].copy()
        mergers_df['segment'] = 'C6'
        mergers_df['match_type'] = 'program flip'

    ## concatenate the two dataframe on the common columns
    combined_matches_c6 = pd.concat([combined_matches, mergers_df], ignore_index=True)
    
    return combined_matches_c6

def find_attritions(combined_matches, dd_df):
    conversions = combined_matches[(combined_matches['match_type'] == 'conversion') & 
                                          (combined_matches['segment'] != 'C2x')].copy()
    pflips = combined_matches[(combined_matches['match_type']=='program flip') & 
                                        (combined_matches['segment'] != 'C2x')].copy()
    attritions = dd_df[~(dd_df['customer_account_id_dd'].isin(conversions['customer_account_id_dd'])) 
                                                                     & ~(dd_df['customer_account_id_dd'].isin(pflips['customer_account_id_dd']))].copy()
    attritions['year'] =  pd.to_datetime(attritions['dd_date']).dt.year
    return attritions

def find_new_accounts(combined_matches, du_df):
    conversions = combined_matches[(combined_matches['match_type'] == 'conversion') & 
                                          (combined_matches['segment'] != 'C2x')].copy()
    pflips = combined_matches[(combined_matches['match_type']=='program flip') & 
                                        (combined_matches['segment'] != 'C2x')].copy()
    new_accounts = du_df[~(du_df['customer_account_id_du'].isin(conversions['customer_account_id_du'])) 
                                                                     & ~(du_df['customer_account_id_du'].isin(pflips['customer_account_id_du']))].copy()
    new_accounts['year'] =  pd.to_datetime(new_accounts['du_date']).dt.year
    return new_accounts

def get_attrition_status(attritions, dd_df, raw_df, 
                        mdm_matches,
                        period_end_date, inactive_period):

    all_ids = set(raw_df['customer_account_id'].unique())
    dd_set = set(dd_df['customer_account_id_dd'])

    ##Non dd Accounts or active accounts
    non_dd = all_ids - dd_set

    ## Exclude the set of customers who are drawing down and find the list of inconsitent customers and customers 
    ## with zero transactions
    ids = list(set(raw_df[~raw_df['customer_account_id'].isin(dd_set)]['customer_account_id']))
    ids_split = split_list(ids, 5)

    ## Get inconsistent customers
    inconsistent_list = []
    for lst in ids_split:
        df = raw_df[(raw_df['customer_account_id'].isin(lst)) & (raw_df['purchase_gallons_qty'] >= 0)].copy()
        df = df[df['purchase_gallons_qty'] >= 0]
        df.reset_index(inplace=True)
        consistent_customers = find_consistent_customers(df, 4)
        inc_list = list(set(lst) - set(consistent_customers))
        inconsistent_list.extend(inc_list)

    ## customer with zero transactions
    gsum = raw_df.groupby(['customer_account_id'])['purchase_gallons_qty'].sum()
    zero_custs = gsum[gsum == 0].index

    ## Combine both the lists
    inconsistent_list.extend(list(zero_custs))
    inconsistent_list = list(set(inconsistent_list))

    mdm_matches = mdm_matches[['accountnumber', 'wex_id', 'partyparentid', 'accountstatus']].copy()
    mdm_matches.rename(columns={'accountstatus':'mdm_status'}, inplace=True)

    attritions.rename(columns={'customer_account_id_dd':'accountnumber',
                                            'year':'dd_year'}, inplace=True)
    mdm_matches = mdm_matches.merge(attritions[['accountnumber', 'dd_date']], how='left')

    inactive_accts = all_ids.intersection(set(inconsistent_list))
    active_accts = all_ids.intersection(non_dd) - inactive_accts

    inactive_accts = inactive_accts.union(all_ids - active_accts)

    frac_status_dic = {k:'Active' for k in active_accts}
    frac_status_dic.update({k: 'Inactive' for k in inactive_accts})
    mdm_matches['frac_status'] = mdm_matches['accountnumber'].map(frac_status_dic)

    ## last 4 month status
    inactive_date_start = pd.to_datetime(period_end_date) + relativedelta(months=-inactive_period+1)
    l4m_active = set(raw_df[(raw_df['revenue_date'] >= inactive_date_start) & 
                            (raw_df['purchase_gallons_qty'] > 0)]['customer_account_id'])

    active_accts_l4m = all_ids.intersection(l4m_active)
    inactive_accts_l4m = all_ids - active_accts_l4m
    l4m_status_dic = {k:'Active' for k in active_accts_l4m}
    l4m_status_dic.update({k: 'Inactive' for k in inactive_accts_l4m})
    mdm_matches['l4m_status'] = mdm_matches['accountnumber'].map(l4m_status_dic)

    mdm_matches.dropna(axis=0, subset=['wex_id'], inplace=True)
    mdm_matches.dropna(axis=0, subset=['partyparentid'], inplace=True)

    mdm_matches['wex_id'] = mdm_matches['wex_id'].astype('int64')
    mdm_matches['partyparentid'] = mdm_matches['partyparentid'].astype('int64')

    mdm_matches.replace({'mdm_status' : {'Suspended' : 'Terminated',
                                            'Converted' : 'Active'}}, inplace=True)
    mdm_matches['frac_status'].fillna('NaN', inplace=True)
    mdm_matches['l4m_status'].fillna('NaN', inplace=True)

    active_w_ids = mdm_matches.groupby(['wex_id']).agg(mdm_status=('mdm_status', set), 
                                                        frac_status=('frac_status', set),
                                                        l4m_status=('l4m_status', set))

    active_w_ids['mdm_status'] = active_w_ids['mdm_status'].apply(lambda x: 'Inactive' if x == {'Terminated'} else 'Active')
    active_w_ids['frac_status'] = active_w_ids['frac_status'].apply(lambda x: 'Inactive' if x == {'Inactive'} else \
                                                                'Active' if 'Active' in x else 'NaN')
    active_w_ids['l4m_status'] = active_w_ids['l4m_status'].apply(lambda x: 'Inactive' if x == {'Inactive'} else \
                                                            'Active' if 'Active' in x else 'NaN')

    active_pp_ids = mdm_matches.groupby(['partyparentid']).agg(mdm_status=('mdm_status', set), 
                                                            frac_status=('frac_status', set),
                                                            l4m_status=('l4m_status', set))

    active_pp_ids['mdm_status'] = active_pp_ids['mdm_status'].apply(lambda x: 'Inactive' if x == {'Terminated'} else 'Active')
    active_pp_ids['frac_status'] = active_pp_ids['frac_status'].apply(lambda x: 'Inactive' if x == {'Inactive'} else \
                                                                'Active' if 'Active' in x else 'NaN')
    active_pp_ids['l4m_status'] = active_pp_ids['l4m_status'].apply(lambda x: 'Inactive' if x == {'Inactive'} else \
                                                            'Active' if 'Active' in x else 'NaN')
    return active_w_ids, active_pp_ids

def get_new_account_status(new_accounts, raw_df, 
                        mdm_matches,
                        period_end_date, inactive_period):
    all_ids = set(raw_df['customer_account_id'].unique())

    mdm_matches = mdm_matches[['accountnumber', 'wex_id', 'partyparentid', 'accountstatus']].copy()
    mdm_matches.rename(columns={'accountstatus':'mdm_status'}, inplace=True)

    ## group by accountnumber first revenue date
    raw_df_awp = raw_df[raw_df['purchase_gallons_qty'] > 0].groupby(['customer_account_id'])['revenue_date'].first()
    raw_df_awp = raw_df_awp.to_frame().reset_index()

    raw_df_awp.rename(columns={'customer_account_id':'accountnumber'}, inplace=True)

    raw_df_awp = raw_df_awp.merge(mdm_matches[['accountnumber', 'wex_id', 'partyparentid']], on=['accountnumber'])
    raw_df_awp.rename(columns={'revenue_date':'first_date'}, inplace=True)
    ## Get the first day of the wex id an dpartyparent id
    wex_id_fd = raw_df_awp.groupby(['wex_id'])['first_date'].min().to_frame().reset_index()
    pp_id_fd = raw_df_awp.groupby(['partyparentid'])['first_date'].min().to_frame().reset_index()

    new_accounts.rename(columns={'customer_account_id_du':'accountnumber'}, inplace=True)
    new_accounts.drop_duplicates(['accountnumber', 'du_date'], inplace=True)

    mdm_matches = mdm_matches.merge(new_accounts[['accountnumber', 'du_date']], 
                                        on=['accountnumber'], 
                                        how='left')

    inactive_date_start = pd.to_datetime(period_end_date) + relativedelta(months=-inactive_period+1)
    l4m_active = set(raw_df[(raw_df['revenue_date'] >= inactive_date_start) & 
                                (raw_df['purchase_gallons_qty'] > 0)]['customer_account_id'])

    active_accts_l4m = all_ids.intersection(l4m_active)
    inactive_accts_l4m = all_ids - active_accts_l4m
    l4m_status_dic = {k:'Active' for k in active_accts_l4m}
    l4m_status_dic.update({k: 'Inactive' for k in inactive_accts_l4m})
    mdm_matches['l4m_status'] = mdm_matches['accountnumber'].map(l4m_status_dic)

    ## Get the first day of the wex id to this dataframe
    mdm_matches.sort_values(['wex_id', 'du_date'], inplace=True)
    mdm_matches_aw = mdm_matches.drop('partyparentid', axis=1).merge(wex_id_fd, on=['wex_id'], how='left')

    ## remove the wex ids
    wex_ids_tbe1 = set(mdm_matches_aw[(mdm_matches_aw['first_date'] < mdm_matches_aw['du_date']) & 
                                    (mdm_matches_aw['first_date'] < pd.to_datetime('2010-06-01'))]['wex_id'])
    wex_ids_tbe2 = set(mdm_matches_aw[(mdm_matches_aw['first_date'].isna())]['wex_id'])

    conditions = [(mdm_matches_aw['du_date'].notna()) & 
                    (~mdm_matches_aw['wex_id'].isin(wex_ids_tbe1.union(wex_ids_tbe2))), 
                    mdm_matches_aw['accountnumber'].isin(all_ids)]
    choices = ['New', 'Old']
        
    mdm_matches_aw['frac_status'] = np.select(conditions, choices, default=np.nan)

    mdm_matches_aw.replace({'mdm_status' : {'Suspended' : 'Terminated',
                                            'Converted' : 'Active'}}, inplace=True)

    new_wex = mdm_matches_aw.groupby(['wex_id']).agg(mdm_status=('mdm_status', set),
                                                    frac_status=('frac_status', set),
                                                    l4m_status=('l4m_status', set),
                                                    first_day=('du_date', min)).reset_index()

    new_wex['mdm_status'] = new_wex['mdm_status'].apply(lambda x: 'Inactive' if x == {'Terminated'} else 'Active')
    new_wex['frac_status'] = new_wex['frac_status'].apply(lambda x: 'New' if x == {'New'} else 'Old' \
                                                        if 'Old' in x else 'NaN')
    new_wex['l4m_status'] = new_wex['l4m_status'].apply(lambda x: 'Inactive' if x == {'Inactive'} else \
                                                        'Active' if 'Active' in x else 'NaN')

    ## Get the first day of the wex id to this dataframe
    mdm_matches.sort_values(['partyparentid', 'du_date'], inplace=True)
    mdm_matches_ap = mdm_matches.drop('wex_id', axis=1).merge(pp_id_fd, on=['partyparentid'], how='left')

    ## remove the wex ids which have a 
    pp_ids_tbe1 = set(mdm_matches_ap[(mdm_matches_ap['first_date'] < mdm_matches_ap['du_date']) & 
                                    (mdm_matches_ap['first_date'] < pd.to_datetime('2010-06-01'))]['partyparentid'])
    pp_ids_tbe2 = set(mdm_matches_ap[(mdm_matches_ap['first_date'].isna())]['partyparentid'])

    conditions = [(mdm_matches_ap['du_date'].notna()) & 
                    (~mdm_matches_ap['partyparentid'].isin(pp_ids_tbe1.union(pp_ids_tbe2))), 
                    mdm_matches_ap['accountnumber'].isin(all_ids)]
    choices = ['New', 'Old']
        
    mdm_matches_ap['frac_status'] = np.select(conditions, choices, default=np.nan)

    mdm_matches_ap.replace({'mdm_status' : {'Suspended' : 'Terminated',
                                            'Converted' : 'Active'}}, inplace=True)

    new_pp = mdm_matches_ap.groupby(['partyparentid']).agg(mdm_status=('mdm_status', set),
                                                    frac_status=('frac_status', set),
                                                    l4m_status=('l4m_status', set),
                                                    first_day=('du_date', min)).reset_index()

    new_pp['mdm_status'] = new_pp['mdm_status'].apply(lambda x: 'Inactive' if x == {'Terminated'} else 'Active')
    new_pp['frac_status'] = new_pp['frac_status'].apply(lambda x: 'New' if x == {'New'} else 'Old' \
                                                        if 'Old' in x else 'NaN')
    new_pp['l4m_status'] = new_pp['l4m_status'].apply(lambda x: 'Inactive' if x == {'Inactive'} else \
                                                        'Active' if 'Active' in x else 'NaN')
    return new_wex, new_pp
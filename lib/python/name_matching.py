from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import Levenshtein
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import re
from re import finditer
from tqdm import tqdm
from tqdm.contrib.itertools import product
from joblib import Parallel, delayed
city_distance_dict = {}
import multiprocessing as mp
cpu_count = int(0.75*mp.cpu_count())

STOP_WORDS = ['inc','llc','co','corp','ltd','limited','in','company','companies','ventures','technologies','works','and','the','of',
              'company','systems','property','express','sons','brothers','building','servic','service','services','construction','constructions',
              'electric','electic','electrical','electronics','groupe','group','solutions','plumbing','enterprises','transport','transportation',
              'systems',',management','contracting','associates','consulting','contrac','contractors','constructors','security','industries',
              'express','sons','properties','investments','investment','corporation','builders','enterprise','store','industrial',
              'automotive','engineering','international','medical','motors','state','communications','communication','delivery',
              'commercial','refrigeration','business','housing','department','technology','foods','productions',
              'manufacturing','contractor','distributors','system','entertainment','hospital','operations','exteriors',
              'associated','foundations','laboratories','black','deck','energy','bus','concrete','control','controls','fire',
              'wheels','test','km','window','cleaning','steemer','dezure','wheels','test','equipment','courier','sanitary',
              'paint','parts','electrica']  

def camel_case_split(s):
    s = str(s)
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', s)
    return ' '.join([m.group(0) for m in matches])

def custom_score(s1,s2):
    s1 = str(s1)
    s2 = str(s2)
    s1 = s1.lower()
    s2 = s2.lower()
    '''
    (total_length - levenshtein distance)*100/total_length
    '''
    ld = Levenshtein.distance(s1, s2)
    total_len = len(s1+s2)
    return int(np.round((1 - ld/total_len)*100,0))

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

def text_preprocessor(df, col=None):
    cleaned = []
    for titles in tqdm(df[col]):
        title = str(titles).lower()
        title = title.strip()
        title = title.replace('\\r', ' ')
        title = title.replace('\\"', ' ')
        title = title.replace('\\n', ' ')
        title = title.replace("'s",'')
        title = re.sub('[^A-Za-z0-9]+', ' ', title)#remove special characters
        if title[0].isnumeric(): #for canadian coutries specific name check it should start with number
            if len(title.split())>=4: #Canadian countries check

                first_numerical_part_in_title = re.findall(r'\d+\.*\d*',title)[0]
                second_third_and_fourth_word = title.split()[1]+' '+title.split()[2]+' '+title.split()[3]
                if len(first_numerical_part_in_title)>=5 and title[:len(first_numerical_part_in_title)].isnumeric() and any(re.findall(r'ontario|alberta|nunavut|canada|british columbia|manitoba|new brunswick|northwest territories|nova scotia|quebec|saskatchewan|yukon|newfoundland labrador|prince edward island', second_third_and_fourth_word, re.IGNORECASE)):
                    pass

                else:
                    title = re.sub('[0-9]+', ' ', title)
            elif len(title.split())==3: #Canadian countries check
                first_numerical_part_in_title = re.findall(r'\d+\.*\d*',title)[0]
                second_and_third_word = title.split()[1]+' '+title.split()[2]
                if len(first_numerical_part_in_title)>=5 and title[:len(first_numerical_part_in_title)].isnumeric() and any(re.findall(r'ontario|alberta|nunavut|canada|british columbia|manitoba|new brunswick|northwest territories|nova scotia|quebec|saskatchewan|yukon', second_and_third_word, re.IGNORECASE)):
                    pass

                else:
                    title = re.sub('[0-9]+', ' ', title)
            elif len(title.split())==2:
                first_numerical_part_in_title = re.findall(r'\d+\.*\d*',title)[0]
                second_word = title.split()[1]
                if len(first_numerical_part_in_title)>=5 and title[:len(first_numerical_part_in_title)].isnumeric() and any(re.findall(r'ontario|alberta|nunavut|manitoba|quebec|saskatchewan|yukon|canada', second_word, re.IGNORECASE)):
                    pass

                else:
                    title = re.sub('[0-9]+', ' ', title)
            else:
                pass
        else:
            title = re.sub('[0-9]+', ' ', title)

        title = ' '.join(f for f in title.split() if f not in STOP_WORDS)
        cleaned.append(title.strip())
    return cleaned

def lvl1_best_match(dd_df, du_df, threshold_score_step1, max_city_distance):
    results_df = pd.DataFrame()
    for i, r in dd_df.iterrows():
        s1_customer_id = r['customer_account_id_dd']
        # print(s1_customer_id)
        s1 = r['customer_account_name_dd']
        unclean_s1 = r['unclean_customer_account_name_dd']
        s1_state = r['account_state_prov_code']
        s1_city = r['account_city']
        s1_drop_date = r['dd_date']

        du_df_loop = du_df.copy()
        
        city_lev_score_list=[]
        for s2_city in list(du_df_loop['account_city']):
            dist = city_distance(s1_city, s2_city)
            city_lev_score_list.append(dist)
        du_df_loop['du_city_fuzzy']=city_lev_score_list
        du_df_loop = du_df_loop[(du_df_loop['du_city_fuzzy'] <= max_city_distance)]

        '''Calculating the Levenshtein score on customer_account_name'''
        score_list=[]
        for s2 in list(du_df_loop['customer_account_name_du']):
            score_list.append(fuzz.partial_ratio(s1, s2))   #partial_ratio
        du_df_loop['du_name_scores']=score_list

        '''Sorting to get top matches in first place'''
        du_df_loop = du_df_loop.sort_values('du_name_scores',ascending = False)
        

        '''If we have more than 10 records ,pick 10th element score and using it as threshold'''
        if len(du_df_loop)>=10:
            lt_10_threshold = du_df_loop['du_name_scores'].iloc[9] #take 10th element's score as threshold
            du_df_loop = du_df_loop[du_df_loop['du_name_scores']>=max(threshold_score_step1, lt_10_threshold)]

        '''Fetching the scores,ids,matches using the indexing performed above'''
        scores = list(du_df_loop['du_name_scores'])
        cty_fzy_scores = list(du_df_loop['du_city_fuzzy'])
        matches_cust_ids = list(du_df_loop['customer_account_id_du'])
        matches = list(du_df_loop['customer_account_name_du'])
        unclean_matches = list(du_df_loop['unclean_customer_account_name_du'])
        state_du = list(du_df_loop['account_state_prov_code'])
        city_du = list(du_df_loop['account_city'])
        # print('city_du', city_du)
        rise_date_du = list(du_df_loop['du_date'])
        # print('rise_date_du', rise_date_du)

        results_dict = defaultdict(list)

        '''Preparing the lists for creating the dataframe'''
        results_dict['customer_id_dd'].append(s1_customer_id)
        results_dict['clean_dd'].append(s1)
        results_dict['unclean_dd'].append(unclean_s1)
        results_dict['state_dd'].append(s1_state)
        results_dict['city_dd'].append(s1_city)
        results_dict['dd_date'].append(s1_drop_date)

        results_dict['customer_id_drawup_matches'].append(matches_cust_ids)
        results_dict['du_matches'].append(matches)
        results_dict['unclean_du_matches'].append(unclean_matches)
        results_dict['state_du'].append(state_du)
        results_dict['city_du'].append(city_du)
        results_dict['du_date'].append(rise_date_du)

        results_dict['scores'].append(scores)
        results_dict['city_scores'].append(cty_fzy_scores)
        results_dict['total_matches_found'].append(len(matches))
        # print(results_dict)
        results_i = pd.DataFrame(results_dict)
        results_df = pd.concat([results_df, results_i])
    return results_df

def level1_match(dd_df, du_df, 
                    match_type=None, 
                    max_city_distance = None,
                    month_diff_h=None,
                    month_diff_l=None,
                    threshold_score_step1=None):
    dd_df = dd_df.copy()
    du_df = du_df.copy()
    if match_type == 'conversion':
        dd_df = dd_df[dd_df['customer_source_system_code'] == 'TANDEM'].copy()
        du_df = du_df[du_df['customer_source_system_code'] == 'SIEBEL'].copy()
    cleaned_dd = text_preprocessor(dd_df, 'customer_account_name_dd')
    cleaned_du = text_preprocessor(du_df, 'customer_account_name_du')

    dd_df.rename(columns={'customer_account_name_dd': 'unclean_customer_account_name_dd'}, inplace=True)
    du_df.rename(columns={'customer_account_name_du': 'unclean_customer_account_name_du'}, inplace=True)

    dd_df['customer_account_name_dd'] = cleaned_dd
    du_df['customer_account_name_du'] = cleaned_du

    dd_df.reset_index(inplace=True)
    du_df.reset_index(inplace=True)

    dd_df.drop('index',inplace=True,axis=1)
    du_df.drop('index',inplace=True,axis=1)

    states = list(dd_df['account_state_prov_code'].unique())
    drop_dates = sorted(list(dd_df['dd_date'].unique()))
    dd_source = ['TANDEM']
    du_source = ['SIEBEL']
    if match_type == 'program_flip':
        dd_source = ['TANDEM', 'SIEBEL']
        du_source = ['TANDEM', 'SIEBEL']

    agg_results = pd.DataFrame()
    #for each combination of state and date
    
    for state, date, dd_s, du_s in product(states, drop_dates, dd_source, du_source):
        # print(state, date, dd_s, du_s)
        if match_type == 'program_flip'  and dd_s != du_s:
            continue

        date = pd.to_datetime(date)

        #fetch drawdown data from that state and date
        dd_df_statewise = dd_df[(dd_df['account_state_prov_code'] == state) &
                                (dd_df['dd_date'] == date) & 
                                (dd_df['customer_source_system_code'] == dd_s)].copy()

        if dd_df_statewise.shape[0] == 0:
            continue
        #setting date thresold(lag of -3months and 12 months)
        start_date = date - pd.DateOffset(months=month_diff_h)
        end_date  = date - pd.DateOffset(months=month_diff_l)
        # print(start_date, end_date)

        #fetch drawup data from that state and date threshold
        du_df_statewise = du_df[(du_df['account_state_prov_code'] == state)].copy()
        du_df_statewise = du_df_statewise[(du_df_statewise['du_date'] >= start_date) &
                                        (du_df_statewise['du_date'] < end_date) & 
                                        (du_df_statewise['customer_source_system_code'] == du_s)].copy()
        # print('du_df_statewise', du_df_statewise.shape)
        if du_df_statewise.shape[0] == 0:
            continue

        results = lvl1_best_match(dd_df_statewise, du_df_statewise, threshold_score_step1, max_city_distance)
        agg_results = pd.concat([agg_results, results])

    return agg_results

def level1_match_mp(dd_df, du_df, 
                    match_type=None, 
                    max_city_distance = None,
                    month_diff_h=None,
                    month_diff_l=None,
                    threshold_score_step1=None):
    dd_df = dd_df.copy()
    du_df = du_df.copy()
    if match_type == 'conversion':
        dd_df = dd_df[dd_df['customer_source_system_code'] == 'TANDEM'].copy()
        du_df = du_df[du_df['customer_source_system_code'] == 'SIEBEL'].copy()
    cleaned_dd = text_preprocessor(dd_df, 'customer_account_name_dd')
    cleaned_du = text_preprocessor(du_df, 'customer_account_name_du')

    dd_df.rename(columns={'customer_account_name_dd': 'unclean_customer_account_name_dd'}, inplace=True)
    du_df.rename(columns={'customer_account_name_du': 'unclean_customer_account_name_du'}, inplace=True)

    dd_df['customer_account_name_dd'] = cleaned_dd
    du_df['customer_account_name_du'] = cleaned_du

    dd_df.reset_index(inplace=True)
    du_df.reset_index(inplace=True)

    dd_df.drop('index',inplace=True,axis=1)
    du_df.drop('index',inplace=True,axis=1)

    states = list(dd_df['account_state_prov_code'].unique())
    drop_dates = sorted(list(dd_df['dd_date'].unique()))
    
    dd_source = ['TANDEM']
    du_source = ['SIEBEL']
    if match_type == 'program_flip':
        dd_source = ['TANDEM', 'SIEBEL']
        du_source = ['TANDEM', 'SIEBEL']

    agg_results = pd.DataFrame()
    #for each combination of state and date
    
    for state, date, dd_s, du_s in product(states, drop_dates, dd_source, du_source):
        # print(state, date, dd_s, du_s)
        if match_type == 'program_flip'  and dd_s != du_s:
            continue

        date = pd.to_datetime(date)

        #fetch drawdown data from that state and date
        dd_df_statewise = dd_df[(dd_df['account_state_prov_code'] == state) &
                                (dd_df['dd_date'] == date) & 
                                (dd_df['customer_source_system_code'] == dd_s)].copy()

        if dd_df_statewise.shape[0] == 0:
            continue
        #setting date thresold(lag of -3months and 12 months)
        start_date = date - pd.DateOffset(months=month_diff_h)
        end_date  = date - pd.DateOffset(months=month_diff_l)

        #fetch drawup data from that state and date threshold
        du_df_statewise = du_df[(du_df['account_state_prov_code'] == state)].copy()
        du_df_statewise = du_df_statewise[(du_df_statewise['du_date'] >= start_date) &
                                        (du_df_statewise['du_date'] < end_date) & 
                                        (du_df_statewise['customer_source_system_code'] == du_s)].copy()
        # print('du_df_statewise', du_df_statewise.shape)
        if du_df_statewise.shape[0] == 0:
            continue

        results = lvl1_best_match_mp(dd_df_statewise, du_df_statewise, threshold_score_step1, max_city_distance)
        agg_results = pd.concat([agg_results, results])

    return agg_results

def tokensortratio_best_match(query_str,
                              search_list,
                              cust_id_list,
                              state_list,
                              city_list,
                              rise_date_list,
                              threshold=None):

    query_str = [query_str]
    #search_list = search_list.strip('][').split(', ')
    '''Creating tuples for string comparison'''
    string_tuples = [(query_str[0],search_list[i]) for i in range(0,len(search_list))]

    '''Calculating the TokenSetRatio score'''
    simple_tokensort=[]
    simple_ld=[]
    treated_tokensort=[]
    for s1, s2 in string_tuples:
        simple_tokensort.append(fuzz.token_sort_ratio(s1, s2))

    for s1, s2 in string_tuples:
        simple_ld.append(custom_score(s1, s2))
    
    for s1, s2 in string_tuples:
        s1 = camel_case_split(s1)
        s2 = camel_case_split(s2)
        treated_tokensort.append(fuzz.token_sort_ratio(s1, s2))

    all_scores = np.array([simple_tokensort, simple_ld, treated_tokensort])
    resultant = list(all_scores.max(axis=0))
    
    # resultant=[]

    # for s1, s2 in string_tuples:
    #     resultant.append(fuzz.token_sort_ratio(s1, s2))

    '''Performing indexing and applying thresholds to get the best possible matches'''
    resultant_dict = dict(zip(range(len(resultant)),resultant))
    resultant_dict_thresholded = {key:val for key, val in resultant_dict.items() if val>=threshold}
    resultant_dict_thresholded_sorted = dict(sorted(resultant_dict_thresholded.items(), key=lambda x: x[1],reverse=True))

    '''Fetching the indices where the best match is'''
    indices = list(resultant_dict_thresholded_sorted.keys())
    scores = list(resultant_dict_thresholded_sorted.values())

    '''Fetching the scores,ids,matches using the indexing performed above'''
    match_customer_id_list = [cust_id_list[i] for i in indices]
    match_list = [search_list[i] for i in indices]
    match_state_list = [state_list[i] for i in indices]
    match_city_list = [city_list[i] for i in indices]
    match_rise_date_list = [rise_date_list[i] for i in indices]
    score_list = [resultant_dict_thresholded_sorted[i] for i in indices]

    sorted_res = []
    for c_id,match,score,state,city,rise_dates in zip(match_customer_id_list,match_list,score_list,match_state_list,match_city_list,match_rise_date_list):
        sorted_res.append((c_id,match,score,state,city,rise_dates))

    return sorted_res

def summary_writer(df):

    global querypoint_list
    global querypoint_cust_ids
    global cust_id_list
    global querypoint_state_list
    global querypoint_city_list
    global querypoint_drop_date_list
    global querypoint_program_name_list
    global match_list
    global score_list
    global state_list
    global city_list
    global rise_date_list
        
    #metric_name = col_name.split('_')[0]
    query_cust_id = df['customer_id_dd']
    query_point = df['unclean_dd']
    query_state = df['state_dd']
    query_city = df['city_dd']
    query_drop_date = df['dd_date']

    row = df['tokensortratio_bestmatch']
    len_row = len(row)
    querypoint_list.extend(np.repeat(query_point,len_row))
    querypoint_cust_ids.extend(np.repeat(query_cust_id,len_row))
    querypoint_state_list.extend(np.repeat(query_state,len_row))
    querypoint_city_list.extend(np.repeat(query_city,len_row))
    querypoint_drop_date_list.extend(np.repeat(query_drop_date,len_row))

    for t in row:
        cust_id_list.append(t[0])
        match_list.append(t[1])
        score_list.append(t[2])
        state_list.append(t[3])
        city_list.append(t[4])
        rise_date_list.append(t[5])

def summary_writer2(df):
    
    summary_df = pd.DataFrame()
    for i,r in tqdm(df.iterrows(), total=df.shape[0]):
        summary_dict = defaultdict(list)
        query_cust_id = r['customer_id_dd']
        query_point = r['unclean_dd']
        query_state = r['state_dd']
        query_city = r['city_dd']
        query_drop_date = r['dd_date']

        row = r['tokensortratio_bestmatch']

        len_row = len(row)
        summary_dict['customer_account_name_dd'].extend(np.repeat(query_point,len_row))
        summary_dict['customer_account_id_dd'].extend(np.repeat(query_cust_id,len_row))
        summary_dict['state_dd'].extend(np.repeat(query_state,len_row))
        summary_dict['city_dd'].extend(np.repeat(query_city,len_row))
        summary_dict['dd_date'].extend(np.repeat(query_drop_date,len_row))

        for t in row:
            summary_dict['customer_account_id_du'].append(t[0])
            summary_dict['customer_account_name_du'].append(t[1])
            summary_dict['name_score'].append(t[2])
            summary_dict['state_du'].append(t[3])
            summary_dict['city_du'].append(t[4])
            summary_dict['du_date'].append(t[5])
        
        summary_i = pd.DataFrame(summary_dict)
        summary_df = pd.concat([summary_df, summary_i])
    return summary_df

def level2_match(agg_results, threshold=None):
    agg_results['tokensortratio_bestmatch'] = agg_results.apply(lambda x: tokensortratio_best_match(x['unclean_dd'],
                                                                                                    x['unclean_du_matches'],
                                                                                                    x['customer_id_drawup_matches'],
                                                                                                    x['state_du'],
                                                                                                    x['city_du'],
                                                                                                    x['du_date'],
                                                                                                    threshold=threshold),
                                                                     axis=1)
    
    global querypoint_list
    global querypoint_cust_ids
    global cust_id_list
    global querypoint_state_list
    global querypoint_city_list
    global querypoint_drop_date_list
    global querypoint_program_name_list
    global match_list
    global score_list
    global state_list
    global city_list
    global rise_date_list
    
    querypoint_list = []
    querypoint_cust_ids = []
    cust_id_list = []
    querypoint_state_list = []
    querypoint_city_list = []
    querypoint_drop_date_list = []
    match_list = []
    score_list = []
    state_list = []
    city_list = []
    rise_date_list = []
    
    ans = agg_results.apply(summary_writer, axis=1)

    # summary_df = summary_writer2(agg_results)

    summary_df = pd.DataFrame()
    summary_df['customer_account_id_dd'] = querypoint_cust_ids
    summary_df['customer_account_name_dd'] = querypoint_list
    summary_df['customer_account_id_du'] = cust_id_list
    summary_df['customer_account_name_du'] = match_list
    summary_df['state_dd'] = querypoint_state_list
    summary_df['city_dd'] = querypoint_city_list
    summary_df['state_du'] = state_list
    summary_df['city_du'] = city_list
    summary_df['dd_date'] = querypoint_drop_date_list
    summary_df['du_date'] = rise_date_list
    summary_df['name_score'] = score_list

    return summary_df

def level1_match_mp_i(r, du_df_statewise, threshold_score_step1, max_city_distance):
    s1_customer_id = r['customer_account_id_dd']
    # print(s1_customer_id)
    s1 = r['customer_account_name_dd']
    unclean_s1 = r['unclean_customer_account_name_dd']
    s1_state = r['account_state_prov_code']
    s1_city = r['account_city']
    s1_drop_date = r['dd_date']

    du_df_statewise_loop = du_df_statewise.copy()

    city_lev_score_list=[]
    for s2_city in list(du_df_statewise_loop['account_city']):
        dist = city_distance(s1_city, s2_city)
        city_lev_score_list.append(dist)
    du_df_statewise_loop['du_city_fuzzy']=city_lev_score_list
    du_df_statewise_loop = du_df_statewise_loop[(du_df_statewise_loop['du_city_fuzzy'] <= max_city_distance)]

    '''Calculating the Levenshtein score on customer_account_name'''
    score_list=[]
    for s2 in list(du_df_statewise_loop['customer_account_name_du']):
        score_list.append(fuzz.partial_ratio(s1, s2))  #partial_ratio
    du_df_statewise_loop['du_name_scores']=score_list

    '''Sorting to get top matches in first place'''
    du_df_statewise_loop = du_df_statewise_loop.sort_values('du_name_scores',ascending = False)

    '''If we have more than 10 records ,pick 10th element score and using it as threshold'''
    if len(du_df_statewise_loop)>=10:
        lt_10_threshold = du_df_statewise_loop['du_name_scores'].iloc[9] #take 10th element's score as threshold
        du_df_statewise_loop = du_df_statewise_loop[du_df_statewise_loop['du_name_scores']>=max(threshold_score_step1, lt_10_threshold)]

    '''Fetching the scores,ids,matches using the indexing performed above'''
    scores = list(du_df_statewise_loop['du_name_scores'])
    cty_fzy_scores = list(du_df_statewise_loop['du_city_fuzzy'])
    matches_cust_ids = list(du_df_statewise_loop['customer_account_id_du'])
    matches = list(du_df_statewise_loop['customer_account_name_du'])
    unclean_matches = list(du_df_statewise_loop['unclean_customer_account_name_du'])
    state_du = list(du_df_statewise_loop['account_state_prov_code'])
    city_du = list(du_df_statewise_loop['account_city'])
    # print('city_du', city_du)
    rise_date_du = list(du_df_statewise_loop['du_date'])
    # print('rise_date_du', rise_date_du)

    results_dict = defaultdict(list)

    '''Preparing the lists for creating the dataframe'''
    results_dict['customer_id_dd'].append(s1_customer_id)
    results_dict['clean_dd'].append(s1)
    results_dict['unclean_dd'].append(unclean_s1)
    results_dict['state_dd'].append(s1_state)
    results_dict['city_dd'].append(s1_city)
    results_dict['dd_date'].append(s1_drop_date)

    results_dict['customer_id_drawup_matches'].append(matches_cust_ids)
    results_dict['du_matches'].append(matches)
    results_dict['unclean_du_matches'].append(unclean_matches)
    results_dict['state_du'].append(state_du)
    results_dict['city_du'].append(city_du)
    results_dict['du_date'].append(rise_date_du)

    results_dict['scores'].append(scores)
    results_dict['city_scores'].append(cty_fzy_scores)
    results_dict['total_matches_found'].append(len(matches))

    results_i = pd.DataFrame(results_dict)

    return results_i

def lvl1_best_match_mp(dd_df, du_df, threshold_score_step1, max_city_distance):
    rows = [dd_df.iloc[i] for i in range(dd_df.shape[0])]
    results = Parallel(n_jobs=cpu_count)(delayed(level1_match_mp_i)(r, du_df, threshold_score_step1, max_city_distance) for r in rows)
    results_df = pd.concat(results).reset_index(drop=True)
    return results_df

def name_match(dd_df, du_df, match_type=None, 
                threshold_score_step1 = 50,
                threshold_score_step2 = 50,
                month_diff_h=6,
                month_diff_l=0,
                max_city_distance = 3):
    '''
    Generates a name matching score between a list of drawdowns and a list of drawups
    
    Parameters
    ----------
        dd_df : pandas.DataFrame
            dataframe containing id, name, state etc of the account which is drawing down
        du_df : pandas.DataFrame
            dataframe containing id, name, state etc of the account which is drawing up
        match_type : str, None
            'conversion' or 'program_flip
        threshold_score : int, 50
            the threshold name score for a match to be retained
        max_city_distance : int, 3
            Max Levenshtein distance between the cities to be considered for a match
    Returns
    -------
        summary_df : pandas.DataFrame
            name matches with its corresponding score
    '''
    lvl1_results = level1_match(dd_df, du_df, 
                                match_type=match_type,
                                threshold_score_step1=threshold_score_step1,
                                month_diff_h=month_diff_h,
                                month_diff_l=month_diff_l,
                                max_city_distance = max_city_distance)
    summary_df = level2_match(lvl1_results, threshold=threshold_score_step2)
    return summary_df

def name_match_mp(dd_df, du_df, match_type=None, 
                threshold_score_step1 = 50,
                threshold_score_step2 = 50,
                month_diff_h=6,
                month_diff_l=0,
                max_city_distance = 3):
    '''
    Generates a name matching score between a list of drawdowns and a list of drawups
    
    Parameters
    ----------
        dd_df : pandas.DataFrame
            dataframe containing id, name, state etc of the account which is drawing down
        du_df : pandas.DataFrame
            dataframe containing id, name, state etc of the account which is drawing up
        match_type : str, None
            'conversion' or 'program_flip
        threshold_score : int, 50
            the threshold name score for a match to be retained
        max_city_distance : int, 3
            Max Levenshtein distance between the cities to be considered for a match
    Returns
    -------
        summary_df : pandas.DataFrame
            name matches with its corresponding score
    '''
    lvl1_results = level1_match_mp(dd_df, du_df, 
                                match_type=match_type,
                                threshold_score_step1=threshold_score_step1,
                                month_diff_h=month_diff_h,
                                month_diff_l=month_diff_l,
                                max_city_distance = max_city_distance)
    summary_df = level2_match(lvl1_results, threshold=threshold_score_step2)
    return summary_df
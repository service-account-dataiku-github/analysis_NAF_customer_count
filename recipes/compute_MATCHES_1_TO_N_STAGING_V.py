# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
from difflib import SequenceMatcher
import Levenshtein

# Read recipe inputs
CALCULATED_DRAW_DOWNS = dataiku.Dataset("CALCULATED_DRAW_DOWNS")
CALCULATED_DRAW_DOWNS_df = CALCULATED_DRAW_DOWNS.get_dataframe()

CALCULATED_DRAW_UPS = dataiku.Dataset("CALCULATED_DRAW_UPS")
CALCULATED_DRAW_UPS_df = CALCULATED_DRAW_UPS.get_dataframe()

COMMON_WORDS = dataiku.Dataset("NAFCUSTOMER_COMMON_WORDS_IN_NAMES")
COMMON_WORDS_df = COMMON_WORDS.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def date_tz_naive(pd_s):
    return pd.to_datetime(pd_s).apply(lambda x:x.tz_localize(None))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# consider customers with the card threshold or more
# set this too low and the running time will balloon

card_cut_off_threshold = 2

df_down_full = CALCULATED_DRAW_DOWNS_df[CALCULATED_DRAW_DOWNS_df.ACTIVE_CARD_MAX>card_cut_off_threshold].copy()
df_up_full = CALCULATED_DRAW_UPS_df[CALCULATED_DRAW_UPS_df.ACTIVE_CARD_MAX>card_cut_off_threshold].copy()

df_down_full.DRAW_DOWN_DATE = date_tz_naive(df_down_full['DRAW_DOWN_DATE'])
df_up_full.DRAW_UP_DATE = date_tz_naive(df_up_full['DRAW_UP_DATE'])

df_up_full.dropna(subset=['DRAW_UP_DATE'], inplace=True)

df_common = COMMON_WORDS_df

df_down_full.sort_values(['CUSTOMER'], inplace=True)
df_up_full.sort_values(['CUSTOMER'], inplace=True)
df_common.sort_values(['WORD'], inplace=True)

print(len(df_down_full), "draw downs full")
print(len(df_up_full), "draw ups full")
print(len(df_common), "common words")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import string

_common_words = df_common.WORD.unique()
print(len(_common_words), "screening against common words")

class Draw_Down_Customer:

    def __init__(self, name, draw_down_date, active_card_max):

        self.CUSTOMER = name
        self.DRAW_DOWN_DATE = draw_down_date
        self.ACTIVE_CARD_MAX = active_card_max

        self.MATCHING_CUSTOMERS = []
        self.PERCENT_DIFFERENCE = []
        self.DAYS_DIFFERENCE = []
        self.DRAW_UP_DATE = []

        # remove punctuation
        c_str = name.translate(str.maketrans('', '', string.punctuation))

        f = c_str.split()
        self.WORD_LIST = []
        for w in f:
            if w not in _common_words:
                self.WORD_LIST.append(w)

    def Match_Draw_Up_Customer(self, name, draw_up_date, active_card_max):

        if (self.CUSTOMER == name):
            # exact match, already captured
            return

        c_str = name.translate(str.maketrans('', '', string.punctuation))

        f = c_str.split()

        check_list = []
        for w in f:
            if (w not in _common_words) and (len(w)>1) and (not w.isnumeric()):
                check_list.append(w)

        percent_diff = round((abs(self.ACTIVE_CARD_MAX - active_card_max) / ((self.ACTIVE_CARD_MAX+active_card_max)/2)),2)

        #date_format = "%Y-%m-%d"
        #d1_date = datetime.strptime(draw_up_date.astype(str), date_format)
        #d2_date = datetime.strptime(self.DRAW_DOWN_DATE.astype(str), date_format)

        delta_between_drop_and_rise = round(abs((draw_up_date-self.DRAW_DOWN_DATE).days)/30.,0)

        for w_to_check in check_list:
            for w in self.WORD_LIST:
                if w_to_check==w:

                    if not name in(self.MATCHING_CUSTOMERS) and(delta_between_drop_and_rise<=4)and(percent_diff<=0.5) :
                        self.MATCHING_CUSTOMERS.append(name)
                        self.PERCENT_DIFFERENCE.append(percent_diff)
                        self.DAYS_DIFFERENCE.append(delta_between_drop_and_rise)
                        self.DRAW_UP_DATE.append(draw_up_date)
                        break;

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import time

def do_save_log(_matching_process_log_time, _matching_process_log_event):

    df_matching_log = pd.DataFrame(_matching_process_log_time)
    if len(df_matching_log)>0:

        df_matching_log.columns = ['LOG_TIME']
        df_matching_log['LOG_EVENT'] = _matching_process_log_event

        MATCHING_PROCESS_LOG_V_df = df_matching_log
        MATCHING_PROCESS_LOG_V = dataiku.Dataset("MATCHING_PROCESS_LOG_V")
        MATCHING_PROCESS_LOG_V.write_with_schema(MATCHING_PROCESS_LOG_V_df)

        print()

def do_save_direct_matches(_direct_customer, _direct_match, _direct_draw_up_date):

    df_matches = pd.DataFrame(_direct_customer)
    if len(df_matches)>0:

        print()
        print("saving", len(df_matches), "1-1 matching records")
        print()

        df_matches.columns = ['CUSTOMER']
        df_matches["MATCH_CUSTOMER"] = _direct_match
        df_matches["DRAW_UP_DATE"] = _direct_draw_up_date

        MATCHES_1_TO_1_STAGING_V_df = df_matches
        MATCHES_1_TO_1_STAGING_V = dataiku.Dataset("MATCHES_1_TO_1_STAGING_V")
        MATCHES_1_TO_1_STAGING_V.write_with_schema(MATCHES_1_TO_1_STAGING_V_df)

        print()

def do_save_multiple_matches(_multiple_customer, _multiple_matches, _multiple_drop_dates):

    df_multiple_matches = pd.DataFrame(_multiple_customer)

    if len(df_multiple_matches)>0:

        print()
        print("saving", len(df_multiple_matches), "1-n matching records")
        print()

        df_multiple_matches.columns = ['CUSTOMER']
        df_multiple_matches["MATCH_CUSTOMER"] = _multiple_matches
        df_multiple_matches["DRAW_UP_DATE"] = _multiple_drop_dates

        MATCHES_1_TO_N_STAGING_V_df = df_multiple_matches
        MATCHES_1_TO_N_STAGING_V = dataiku.Dataset("MATCHES_1_TO_N_STAGING_V")
        MATCHES_1_TO_N_STAGING_V.write_with_schema(MATCHES_1_TO_N_STAGING_V_df)

        print()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from datetime import timedelta

df_down = df_down_full
df_up = df_up_full

_processed_customers = []
verbose = False

_matching_process_log_time = []
_matching_process_log_event = []

_direct_customer = []
_direct_match = []
_direct_draw_up_date = []

_multiple_customer = []
_multiple_matches = []
_multiple_drop_dates = []

_no_match_customer = []

save_every_n = 50
to_save_counter = 0
print_every_n = 100

print(len(df_down), "filtered down rows")
print(len(df_up), "filtered up rows")

_customers = []

t0 = time.time()

for index, row in df_down.iterrows():

    customer = row['CUSTOMER']
    draw_down_date = row['DRAW_DOWN_DATE']
    active_card_max = row['ACTIVE_CARD_MAX']

    c = Draw_Down_Customer(customer, draw_down_date, active_card_max)

    _customers.append(c)

idx = 0

_matching_process_log_time.append(str(pd.Timestamp.now()))
_matching_process_log_event.append(" processing range " + str(len(_customers)) + " Draw Down Customers")
do_save_log(_matching_process_log_time, _matching_process_log_event)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
idx=0
for c in _customers:

    idx+=1

    date_start = pd.to_datetime(c.DRAW_DOWN_DATE) +timedelta(days=-120)
    date_end = pd.to_datetime(c.DRAW_DOWN_DATE) +timedelta(days=120)

    card_delta = c.ACTIVE_CARD_MAX * 0.5
    card_start = c.ACTIVE_CARD_MAX - card_delta
    card_end = c.ACTIVE_CARD_MAX + card_delta

    df_up = df_up_full[(df_up_full.ACTIVE_CARD_MAX>=card_start)&
                   (df_up_full.ACTIVE_CARD_MAX<=card_end)&
                    (df_up_full.DRAW_UP_DATE >= pd.to_datetime(date_start))&
                  (df_up_full.DRAW_UP_DATE <= pd.to_datetime(date_end))].copy()

    df_up['distance'] = 0.0
    df_up.distance = df_up.apply(lambda x: Levenshtein.ratio(x['CUSTOMER'],c.CUSTOMER),axis=1)
    df_up.dropna(subset=['distance'], inplace=True)

    df_up = df_up[df_up.distance>0.8]

    for index_up, row_up in df_up.iterrows():

        customer = row_up['CUSTOMER']
        draw_up_date = row_up['DRAW_UP_DATE']
        active_card_max = row_up['ACTIVE_CARD_MAX']

        c.Match_Draw_Up_Customer(customer, draw_up_date, active_card_max)

    if len(c.MATCHING_CUSTOMERS)==1:

        if not c.CUSTOMER in (_processed_customers):

            to_save_counter += 1

            _direct_customer.append(c.CUSTOMER)
            _processed_customers.append(c.CUSTOMER)
            _direct_match.append(c.MATCHING_CUSTOMERS[0])
            _processed_customers.append(c.MATCHING_CUSTOMERS[0])
            _direct_draw_up_date.append(c.DRAW_UP_DATE[0])

            if verbose:
                print()
                print("DIRECT")
                print(c.CUSTOMER, c.WORD_LIST)
                print(c.MATCHING_CUSTOMERS)
                print(c.PERCENT_DIFFERENCE)
                print(c.DAYS_DIFFERENCE)
                print("=====")
                print()

    elif len(c.MATCHING_CUSTOMERS)>1:

        if not c.CUSTOMER in (_processed_customers):

            to_save_counter += 1

            _multiple_customer.append(c.CUSTOMER)
            _processed_customers.append(c.CUSTOMER)
            _multiple_matches.append(c.MATCHING_CUSTOMERS)
            _multiple_drop_dates.append(c.DRAW_UP_DATE)

        if verbose:
            print()
            print("MULTIPLE")
            print(c.CUSTOMER, c.WORD_LIST)
            print(c.MATCHING_CUSTOMERS)
            print(c.PERCENT_DIFFERENCE)
            print(c.DAYS_DIFFERENCE)
            print("=====")
            print()

    else:

        # could not find a match, remove it from future processing
        _no_match_customer.append(c.CUSTOMER)
        _processed_customers.append(c.CUSTOMER)

    if to_save_counter>=save_every_n:

        _matching_process_log_time.append(str(pd.Timestamp.now()))
        _matching_process_log_event.append("writing datasets to snowflake")
        do_save_log(_matching_process_log_time, _matching_process_log_event)

        do_save_direct_matches(_direct_customer, _direct_match, _direct_draw_up_date)
        do_save_multiple_matches(_multiple_customer, _multiple_matches, _multiple_drop_dates)

        _matching_process_log_time.append(str(pd.Timestamp.now()))
        _matching_process_log_event.append("saved " + str(to_save_counter) + " records to snowflake.")
        do_save_log(_matching_process_log_time, _matching_process_log_event)

        to_save_counter = 0

    t1 = time.time()

    avg_duration = (((t1-t0)/idx)/60.0)

    if idx % print_every_n == 0:
        idx_remaining = len(_customers)-idx
        print("processing", idx, "current record:", c.CUSTOMER, ",", idx_remaining, "remaining")
        print(round(avg_duration,2), "avg mins per iteration",  round((avg_duration*idx_remaining)/60,2), "estimated hrs remaining")
        print(len(_direct_customer), "direct match records", len(_multiple_customer), "multiple match records", len(_no_match_customer), "no match records")
        print()

_matching_process_log_time.append(str(pd.Timestamp.now()))
_matching_process_log_event.append("writing datasets to snowflake")
do_save_log(_matching_process_log_time, _matching_process_log_event)

do_save_direct_matches(_direct_customer, _direct_match, _direct_draw_up_date)
do_save_multiple_matches(_multiple_customer, _multiple_matches, _multiple_drop_dates)

_matching_process_log_time.append(str(pd.Timestamp.now()))
_matching_process_log_event.append("saved " + str(to_save_counter) + " records to snowflake.")
do_save_log(_matching_process_log_time, _matching_process_log_event)
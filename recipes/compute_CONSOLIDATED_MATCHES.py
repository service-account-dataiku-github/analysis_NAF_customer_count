# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
CALCULATED_CARD_DRAW_UPS = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS")
CALCULATED_CARD_DRAW_UPS_df = CALCULATED_CARD_DRAW_UPS.get_dataframe()

CALCULATED_CARD_DRAW_UPS_FULL = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS_FULL")
CALCULATED_CARD_DRAW_UPS_FULL_df = CALCULATED_CARD_DRAW_UPS_FULL.get_dataframe()

CALCULATED_CARD_DRAW_DOWNS = dataiku.Dataset("CALCULATED_CARD_DRAW_DOWNS")
CALCULATED_CARD_DRAW_DOWNS_df = CALCULATED_CARD_DRAW_DOWNS.get_dataframe()

CALCULATED_CARD_DRAW_DOWNS_FULL = dataiku.Dataset("CALCULATED_CARD_DRAW_DOWNS_FULL")
CALCULATED_CARD_DRAW_DOWNS_FULL_df = CALCULATED_CARD_DRAW_DOWNS_FULL.get_dataframe()

COMMON_WORDS = dataiku.Dataset("NAFCUSTOMER_COMMON_WORDS_IN_NAMES")
COMMON_WORDS_df = COMMON_WORDS.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_down = CALCULATED_CARD_DRAW_DOWNS_df
df_down_full = CALCULATED_CARD_DRAW_DOWNS_FULL_df
df_up = CALCULATED_CARD_DRAW_UPS_df
df_up_full = CALCULATED_CARD_DRAW_UPS_FULL_df
df_common = COMMON_WORDS_df

df_down.sort_values(['CUSTOMER'], inplace=True)
df_down_full.sort_values(['CUSTOMER'], inplace=True)
df_up.sort_values(['CUSTOMER'], inplace=True)
df_up_full.sort_values(['CUSTOMER'], inplace=True)
df_common.sort_values(['WORD'], inplace=True)

print(len(df_down), "draw downs")
print(len(df_down_full), "draw downs full")
print(len(df_up), "draw ups")
print(len(df_up_full), "draw ups full")
print(len(df_common), "common words")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_down.head()
df_common.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import string

_common_words = df_common.WORD.unique()
print(len(_common_words), "screening against common words")

class Draw_Down_Customer:

    def __init__(self, name, draw_down_date, mean_dd, std_dd, active_card_max):

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

    def Match_Draw_Up_Customer(self, name, draw_up_date, mean_du, std_du, active_card_max):

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
idx = 0
_customers = []
verbose = True

df_down = df_down_full[df_down_full.ACTIVE_CARD_MAX>1000]
df_up = df_up_full[df_up_full.ACTIVE_CARD_MAX>1000]

print(len(df_down), "filtered down rows")
print(len(df_up), "filtered up rows")

max_idx = 1000

for index, row in df_down.iterrows():

    idx+=1

    customer = row['CUSTOMER']
    draw_down_date = row['DRAW_DOWN_DATE']
    mean_dd = row['MEAN_DD']
    std_dd = row['STD_DD']
    active_card_max = row['ACTIVE_CARD_MAX']

    c = Draw_Down_Customer(customer, draw_down_date, mean_dd, std_dd, active_card_max)

    _customers.append(c)

    if max_idx>0:
        if idx>max_idx:
            break;

idx = 0

_direct_customer = []
_direct_match = []
_direct_draw_up_date = []

_multiple_customer = []
_multiple_matches = []
_multiple_drop_dates = []

_no_match_customer = []

for c in _customers:

    for index_up, row_up in df_up.iterrows():

        idx+=1

        customer = row_up['CUSTOMER']
        draw_up_date = row_up['DRAW_UP_DATE']
        mean_du = row_up['MEAN_DU']
        std_du = row_up['STD_DU']
        active_card_max = row_up['ACTIVE_CARD_MAX']

        c.Match_Draw_Up_Customer(customer, draw_up_date, mean_du, std_du, active_card_max)

    if len(c.MATCHING_CUSTOMERS)==1:

        _direct_customer.append(c.CUSTOMER)
        _direct_match.append(c.MATCHING_CUSTOMERS[0])
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

        _multiple_customer.append(c.CUSTOMER)
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
        _no_match_customer.append(c.CUSTOMER)

print(idx)
print()

print(len(_direct_customer), "direct matches")
print(len(_multiple_matches), "multiple matches")
print(len(_no_match_customer), "no match customers")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_matches = pd.DataFrame(_direct_customer)
df_matches.columns = ['CUSTOMER']
df_matches["MATCH_CUSTOMER"] = _direct_match
df_matches["DRAW_UP_DATE"] = _direct_draw_up_date
df_matches.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_multiple_matches = pd.DataFrame(_multiple_customer)
df_multiple_matches.columns = ['CUSTOMER']
df_multiple_matches["MATCH_CUSTOMER"] = _multiple_matches
df_multiple_matches["DRAW_UP_DATE"] = _multiple_drop_dates
df_multiple_matches.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
MATCHES_1_TO_N_FOR_MANUAL_REVIEW_df = df_multiple_matches
MACTHES_1_TO_N_FOR_MANUAL_REVIEW = dataiku.Dataset("MATCHES_1_TO_N_FOR_MANUAL_REVIEW")
MACTHES_1_TO_N_FOR_MANUAL_REVIEW.write_with_schema(MATCHES_1_TO_N_FOR_MANUAL_REVIEW_df)

MATCHES_1_TO_1_df = df_matches
MATCHES_1_TO_1 = dataiku.Dataset("MATCHES_1_TO_1")
MATCHES_1_TO_1.write_with_schema(MATCHES_1_TO_1_df)
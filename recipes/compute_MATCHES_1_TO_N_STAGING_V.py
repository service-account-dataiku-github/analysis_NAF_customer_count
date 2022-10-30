# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
CALCULATED_DRAW_DOWNS = dataiku.Dataset("CALCULATED_DRAW_DOWNS")
CALCULATED_DRAW_DOWNS_df = CALCULATED_DRAW_DOWNS.get_dataframe()

CALCULATED_DRAW_UPS = dataiku.Dataset("CALCULATED_DRAW_UPS")
CALCULATED_DRAW_UPS_df = CALCULATED_DRAW_UPS.get_dataframe()

COMMON_WORDS = dataiku.Dataset("NAFCUSTOMER_COMMON_WORDS_IN_NAMES")
COMMON_WORDS_df = COMMON_WORDS.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_down_full = CALCULATED_DRAW_DOWNS_df
df_up_full = CALCULATED_DRAW_UPS_df
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
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

#MATCHES_1_TO_N_STAGING_V_df = ... # Compute a Pandas dataframe to write into MATCHES_1_TO_N_STAGING_V
#MATCHES_1_TO_1_STAGING_V_df = ... # Compute a Pandas dataframe to write into MATCHES_1_TO_1_STAGING_V


# Write recipe outputs
#MATCHES_1_TO_N_STAGING_V = dataiku.Dataset("MATCHES_1_TO_N_STAGING_V")
#MATCHES_1_TO_N_STAGING_V.write_with_schema(MATCHES_1_TO_N_STAGING_V_df)
#MATCHES_1_TO_1_STAGING_V = dataiku.Dataset("MATCHES_1_TO_1_STAGING_V")
#MATCHES_1_TO_1_STAGING_V.write_with_schema(MATCHES_1_TO_1_STAGING_V_df)
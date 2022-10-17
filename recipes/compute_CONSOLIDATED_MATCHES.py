# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
CALCULATED_CARD_DRAW_UPS = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS")
CALCULATED_CARD_DRAW_UPS_df = CALCULATED_CARD_DRAW_UPS.get_dataframe()

CALCULATED_CARD_DRAW_DOWNS = dataiku.Dataset("CALCULATED_CARD_DRAW_DOWNS")
CALCULATED_CARD_DRAW_DOWNS_df = CALCULATED_CARD_DRAW_DOWNS.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_down = CALCULATED_CARD_DRAW_DOWNS_df
df_up = CALCULATED_CARD_DRAW_UPS_df

df_down.sort_values(['CUSTOMER'], inplace=True)
df_up.sort_values(['CUSTOMER'], inplace=True)

print(len(df_down), "draw downs")
print(len(df_up), "draw ups")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_down.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import string

class Draw_Down_Customer:
    
    def __init__(self, name, draw_down_date, mean_dd, std_dd, active_card_max):

        self._common_words = ['PIZZA', 'MANAGEMENT', 'THERAPEUTICS', 'USA', 'INC', 'US', 'EQUIPMENT', 'MEDICAL', 'SYSTEMS',
                             'ANIMAL', 'HEALTH', 'LLC', 'CORPORATION', 'BRANDS', 'TIRE', 'RUBBER', 'COUNTRY', 'CORP', 
                              'PHARMACY','INC', 'RESTAURANTS', 'CONTAINER', 'AMERICA', 'APPLICATIONS', 'TECHNOLOGY', 
                              'INSURANCE', 'FARM','CREDIT', 'SERVICES', 'SERVICE', 'ACCOUNT', 'GENERAL', 'PARTS', 
                              'INTL', 'FLAVORS', 'HOLDINGS', 'FOOD','INDUSTRIES', 'LP', 'FLEET', 'MEDICAL', 'PHARMA',
                             'GLOBAL', 'PIPELINE', 'WHEELS', 'BIOSCIENCES', 'SSI', 'SPRINGS', 'NORTH', 'MARINE', 'HOLDING', 
                              'TECHNOLOGIES','GROUP', 'PHARMACEUTICAL', 'NA', 'USA', 'COMPANY', 'RAIL', 'PARTNERS', 'BROS', 
                              'CO', 'PHARMACEUTICALS', 'ENERGY', 'DISTRIBUTION', 'DENTAL', 'SPECIALTIES', 'OPERATIONS', 
                              'COMPANY', 'THE', 'MOUNTAIN', 'TRANS', 'FUEL', 'AMERICAN', 'HOMES', 'GAS']
        
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
            if w not in self._common_words:
                self.WORD_LIST.append(w)
                
    def Match_Draw_Up_Customer(self, name, draw_up_date, mean_du, std_du, active_card_max):
        
        if (self.CUSTOMER == name):
            # exact match, already captured
            return
        
        c_str = name.translate(str.maketrans('', '', string.punctuation))
        
        f = c_str.split()
        
        check_list = []
        for w in f:
            if w not in self._common_words:
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

for index, row in df_down.iterrows():
    
    idx+=1 
    
    customer = row['CUSTOMER']
    draw_down_date = row['DRAW_DOWN_DATE']
    mean_dd = row['MEAN_DD']
    std_dd = row['STD_DD']
    active_card_max = row['ACTIVE_CARD_MAX']
    
    c = Draw_Down_Customer(customer, draw_down_date, mean_dd, std_dd, active_card_max)
    
    _customers.append(c)
    
    #if idx>10:
    #    break;
    
idx = 0
verbose = False

_direct_customer = []
_direct_match = []
_direct_draw_up_date = []


_direct_matches = []
_multiple_matches = []

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
        _direct_match.append(c.c.MATCHING_CUSTOMERS[0])
        _direct_draw_up_date(c.c.DRAW_UP_DATE[0])
        
        print(c.CUSTOMER, c.MATCHING_CUSTOMERS)
        
        if verbose:
            
            print(c.CUSTOMER, c.WORD_LIST)
            print("Draw Up Date:", c.DRAW_DOWN_DATE)
            print("Cards", c.ACTIVE_CARD_MAX)
            print()
            print(c.MATCHING_CUSTOMERS)
            print(c.PERCENT_DIFFERENCE)
            print(c.DAYS_DIFFERENCE)
            print()
            print("=====")
            print()
        
    elif len(c.MATCHING_CUSTOMERS)>1:
        
        _multiple_matches.append([c.CUSTOMER, c.WORD_LIST])
        
        if verbose:
            print()
            print("deal with multiple matches")
            print()
        
print(idx)
print()

print(len(_direct_matches), "direct matches")
print(len(_multiple_matches), "multiple matches")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(df_up)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

#CONSOLIDATED_MATCHES_df = ... # Compute a Pandas dataframe to write into CONSOLIDATED_MATCHES


# Write recipe outputs
#CONSOLIDATED_MATCHES = dataiku.Dataset("CONSOLIDATED_MATCHES")
#CONSOLIDATED_MATCHES.write_with_schema(CONSOLIDATED_MATCHES_df)
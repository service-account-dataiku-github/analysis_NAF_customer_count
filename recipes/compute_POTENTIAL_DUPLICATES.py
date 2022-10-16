# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
CALCULATED_CARD_DRAW_DOWNS = dataiku.Dataset("CALCULATED_CARD_DRAW_DOWNS")
CALCULATED_CARD_DRAW_DOWNS_df = CALCULATED_CARD_DRAW_DOWNS.get_dataframe()

CALCULATED_CARD_DRAW_UPS = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS")
CALCULATED_CARD_DRAW_UPS_df = CALCULATED_CARD_DRAW_UPS.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_up = CALCULATED_CARD_DRAW_UPS_df
print(len(df_up))
df_up.DRAW_UP_DATE = pd.to_datetime(df_down.DRAW_DOWN_DATE).dt.date
df_up.sort_values(['CUSTOMER'], inplace=True)

df_down = CALCULATED_CARD_DRAW_DOWNS_df
print(len(df_down))
df_down.DRAW_DOWN_DATE = pd.to_datetime(df_down.DRAW_DOWN_DATE).dt.date
df_down.sort_values(['CUSTOMER'], inplace=True)
df_up.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from datetime import datetime

# ===============================================
class Draw_Down_Customer:
    
    def __init__(self, name, draw_down_date, mean_dd, std_dd, active_card_max):

        self._common_words = ['PIZZA', 'MANAGEMENT', 'USA', 'INC', 'US', 'EQUIPMENT', 'MEDICAL', 'SYSTEMS',
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
                
        percent_diff = (abs(self.ACTIVE_CARD_MAX - active_card_max) / ((self.ACTIVE_CARD_MAX+active_card_max)/2))*100.

        #date_format = "%Y-%m-%d"
        #d1_date = datetime.strptime(draw_up_date.astype(str), date_format)
        #d2_date = datetime.strptime(self.DRAW_DOWN_DATE.astype(str), date_format)

        #delta = d2_date-d1_date
        
        if 
        
            
        for w_to_check in check_list:
            for w in self.WORD_LIST:
                if w_to_check==w:
                    if not name in(self.MATCHING_CUSTOMERS):
                        self.MATCHING_CUSTOMERS.append(name)
                        self.PERCENT_DIFFERENCE.append(percent_diff)
                        self.DAYS_DIFFERENCE.append(draw_up_date)
                        break;
       

idx = 0

_draw_down_customers = []
        
for index, row in df_down.iterrows():
    
    idx+=1
    customer = row['CUSTOMER']
    draw_down_date = row['DRAW_DOWN_DATE']
    mean_dd = row['MEAN_DD']
    std_dd = row['STD_DD']
    active_card_max = row['ACTIVE_CARD_MAX']
    
    c = Draw_Down_Customer(customer, draw_down_date, mean_dd, std_dd, active_card_max)
    _draw_down_customers.append(c)
    
    #if idx>5:
    #    break;
        
for c in _draw_down_customers:
    
    for index_up, row_up in df_up.iterrows():
        
        customer = row_up['CUSTOMER']
        draw_up_date = row_up['DRAW_UP_DATE']
        mean_du = row_up['MEAN_DU']
        std_du = row_up['STD_DU']
        active_card_max = row_up['ACTIVE_CARD_MAX']
        
        c.Match_Draw_Up_Customer(customer, draw_up_date, mean_du, std_du, active_card_max)

match_count = 0
for c in _draw_down_customers:
    
    if len(c.MATCHING_CUSTOMERS)>0:
        match_count+=1
        print(c.CUSTOMER)
        print(c.MATCHING_CUSTOMERS)
        print(c.PERCENT_DIFFERENCE)
        print(c.DAYS_DIFFERENCE)
        print()
        
print(match_count)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_up.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
customer_down_list = CALCULATED_CARD_DRAW_DOWNS_df.CUSTOMER.unique()
customer_up_list = CALCULATED_CARD_DRAW_UPS_df.CUSTOMER.unique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
CALCULATED_CARD_DRAW_DOWNS_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

POTENTIAL_DUPLICATES_df = ... # Compute a Pandas dataframe to write into POTENTIAL_DUPLICATES


# Write recipe outputs
POTENTIAL_DUPLICATES = dataiku.Dataset("POTENTIAL_DUPLICATES")
POTENTIAL_DUPLICATES.write_with_schema(POTENTIAL_DUPLICATES_df)
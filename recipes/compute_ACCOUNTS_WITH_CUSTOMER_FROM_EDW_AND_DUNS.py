# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import string

# Read recipe inputs
ACCOUNTS_WITH_BUNDLER_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_BUNDLER_AND_DUNS")
ACCOUNTS_WITH_BUNDLER_AND_DUNS_df = ACCOUNTS_WITH_BUNDLER_AND_DUNS.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = ACCOUNTS_WITH_BUNDLER_AND_DUNS_df

import warnings
warnings.filterwarnings(action='once')

df['DUNS'] = df['DUNS'].astype('Int64', errors='ignore')
df['DNB_DUNS_NUMBER'] = df['DNB_DUNS_NUMBER'].astype('Int64', errors='ignore')
df['DNB_BUSINESS_NAME'] = df['DNB_BUSINESS_NAME'].str.upper()
df["DNB_BUSINESS_NAME"] = df['DNB_BUSINESS_NAME'].str.replace('[^\w\s]','')

df['DNB_GLOBAL_ULT_NUMBER'] = df['DNB_GLOBAL_ULT_NUMBER'].astype('Int64', errors='ignore')
df['DNB_GLOBAL_ULT_NAME'] = df['DNB_GLOBAL_ULT_NAME'].str.upper()
df["DNB_GLOBAL_ULT_NAME"] = df['DNB_GLOBAL_ULT_NAME'].str.replace('[^\w\s]','')

df['DNB_DOMESTIC_ULT_NUMBER'] = df['DNB_DOMESTIC_ULT_NUMBER'].astype('Int64', errors='ignore')
df['DNB_DOMESTIC_ULTIMATE_NAME'] = df['DNB_DOMESTIC_ULTIMATE_NAME'].str.upper()
df["DNB_DOMESTIC_ULTIMATE_NAME"] = df['DNB_DOMESTIC_ULTIMATE_NAME'].str.replace('[^\w\s]','')

df['DNB_HQ_NUMBER'] = df['DNB_HQ_NUMBER'].astype('Int64', errors='ignore')
df['DNB_HQ_NAME'] = df['DNB_HQ_NAME'].str.upper()
df["DNB_HQ_NAME"] = df['DNB_HQ_NAME'].str.replace('[^\w\s]','')

df['DNB_LEVEL'] = 'None'
df['DNB_CUSTOMER_NAME'] = np.nan

df.loc[~df["DNB_GLOBAL_ULT_NAME"].isnull(),'DNB_LEVEL'] = "DUNS Global"
df.loc[~df["DNB_GLOBAL_ULT_NAME"].isnull(),'DNB_CUSTOMER_NAME'] = df.DNB_GLOBAL_ULT_NAME

df.loc[(df["DNB_GLOBAL_ULT_NAME"].isnull())&(~df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull()),'DNB_LEVEL'] = "DUNS Domestic"
df.loc[(df["DNB_GLOBAL_ULT_NAME"].isnull())&(~df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_DOMESTIC_ULTIMATE_NAME

df.loc[(df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull())&(~df["DNB_HQ_NAME"].isnull()),'DNB_LEVEL'] = "DUNS HQ"
df.loc[(df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull())&(~df["DNB_HQ_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_HQ_NAME

df.loc[(df["DNB_HQ_NAME"].isnull())&(~df["DNB_BUSINESS_NAME"].isnull()),'DNB_LEVEL'] = "DUNS"
df.loc[(df["DNB_HQ_NAME"].isnull())&(~df["DNB_BUSINESS_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_BUSINESS_NAME

df['EDW_STATE'] = 'Unknown'
df.loc[df["EDW_CUSTOMER_NAME"].isnull(),'EDW_STATE'] = "None"
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'EDW_STATE'] = "Set"

df['CUSTOMER'] = np.nan
df['CUST_CALC_SOURCE'] = 'Unknown'
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'CUSTOMER'] = df["EDW_CUSTOMER_NAME"]
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'CUST_CALC_SOURCE'] = "EDW"

df.loc[(df["CUSTOMER"].isnull())&(~df["DNB_CUSTOMER_NAME"].isnull()),'CUST_CALC_SOURCE'] = "DNB"
df.loc[df["CUSTOMER"].isnull(),'CUSTOMER'] = df["DNB_CUSTOMER_NAME"]

df.loc[df["CUSTOMER"].isnull(),'CUST_CALC_SOURCE'] = 'ACCOUNT'
df.loc[df["CUSTOMER"].isnull(),'CUSTOMER'] = df["CUSTOMER_ACCOUNT_NAME"]

# RULE SETs
def apply_rule(df, rule_name,filter_name_list,final_name):

    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUST_CALC_SOURCE"] = rule_name
    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUSTOMER"] = final_name

    return(df)

# this set of rules represents high card count mappings
# these rules have been manually verified
# as they impact large numbers of cards (and in turn gallons, spend and revenue)
df = apply_rule(df, "RULE 001", ['QUANTA SERVICES INC','QUANTA SERVICES'], 'QUANTA SERVICES INC')
df = apply_rule(df, "RULE 002", ['0113 WINDSTREAM COMM','0113 WINDSTREAM COMM (2)'], '0113 WINDSTREAM COMM')
df = apply_rule(df, "RULE 003", ['1033 MONSANTO COMPANY (25)','1033 MONSANTO COMPANY (32)'], '1033 MONSANTO COMPANY')
df = apply_rule(df, "RULE 004", ['2536 HOME DEPOT','2536 HOME DEPOT 5'], '2536 HOME DEPOT')
df = apply_rule(df, "RULE 005", ['3274 MEDTRONIC 2','3274 MEDTRONIC','3274 MEDTRONIC AD'], '3274 MEDTRONIC')
df = apply_rule(df, "RULE 006", ['3373 BASF','3373 BASF AD'], '3373 BASF')
df = apply_rule(df, "RULE 007", ['5929-TESLA (2)','5929-TESLA','5929-TESLA (3)'], '5929-TESLA')

df = apply_rule(df, "RULE 008", ['6220-KONE INC','6220-KONE INC','6220-KONE INC (3)'], '6220-KONE INC')
df = apply_rule(df, "RULE 009", ['7325 ADVANCE STORES COMPANY 4','7325 ADVANCE AUTO','7325 ADVANCE STORES COMP','7325 ADVANCE STORES COMP 2'], '')
df = apply_rule(df, "RULE 010", ['17435-GE HEALTHCARE (3)','17435 GE HEALTHCARE','17435-GE HEALTHCARE','17435-GE HEALTHCARE (2)'], '17435-GE HEALTHCARE')
df = apply_rule(df, "RULE 011", ['17595-SCHLUMBERGER','17595-SCHLUMBERGER','17595-SCHLUMBERGER (2)','17595 SCHLUMBERGER AD'], '17595 SCHLUMBERGER')
df = apply_rule(df, "RULE 012", ['ADAPTHEALTH CORP','ADAPTHEALTH LLC'], 'ADAPTHEALTH')
df = apply_rule(df, "RULE 013", ['AGILENT TECHNOLOGIES LP','AGILENT TECHNOLOGIES LP 2'], 'AGILENT TECHNOLOGIES LP')
df = apply_rule(df, "RULE 014", ['AGL RESOURCES 5CU6','AGL RESOURCES ARI'], 'AGL RESOURCES')
df = apply_rule(df, "RULE 015", ['APTIVE ENVIRONMENTAL LLC','ADAPTIVE ENVIRONMENTAL CONSULTING INC','APTIVE ENVIRONMENTAL LLC','APTIVE ENVIRONMENTAL 0CV7'], 'APTIVE ENVIRONMENTAL')
df = apply_rule(df, "RULE 016", ['ARAMARK (RPT PARENT)','ARAMARK RPT PARENT','ARAMARK FOOD & FACILITY HYBRID','ARAMARK','ARAMARK 2G93'], 'ARAMARK')
df = apply_rule(df, "RULE 017", ['CABLE ONE (5R09)','CABLE ONE INC 5R09','CABLE ONE INC'], 'CABLE ONE INC')
df = apply_rule(df, "RULE 018", ['CBRE 3','CBRE','CBRE 2','CBRE GROUP INC'], 'CBRE GROUP INC')
df = apply_rule(df, "RULE 019", ['COMCAST CABLE','COMCAST CABLE HQ 5BA6','COMCAST CABLE WEST 5BA6','COMCAST CABLE NE 5BA6','COMCAST CABLE CE 5BA6','COMCAST CABLE COMMUN 5FH6'], 'COMCAST CABLE')
df = apply_rule(df, "RULE 020", ['COMPASS GROUP','COMPASS GROUP PLC','COMPASS GROUP 5968'], 'COMPASS GROUP')
df = apply_rule(df, "RULE 021", ['CONOCOPHILLIPS','CONOCOPHILLIPS COMPANY','CONOCOPHILLIPS CO'], 'CONOCOPHILLIPS')
#df = apply_rule(df, "RULE 017", ['',''], '')
#df = apply_rule(df, "RULE 017", ['',''], '')
#df = apply_rule(df, "RULE 017", ['',''], '')


#df = apply_rule(df, "RULE 000", ['',''], '')

print(len(df))
df.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df = df

# Write recipe outputs
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS")
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS.write_with_schema(ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df)
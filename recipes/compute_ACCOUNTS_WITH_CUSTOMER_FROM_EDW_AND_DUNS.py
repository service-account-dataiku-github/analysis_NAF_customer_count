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

df = apply_rule(df, "RULE 000", ['6220-KONE INC','6220-KONE INC','6220-KONE INC (3)'], '6220-KONE INC')
df = apply_rule(df, "RULE 000", ['7325 ADVANCE STORES COMPANY 4','7325 ADVANCE AUTO','7325 ADVANCE STORES COMP','7325 ADVANCE STORES COMP 2'], '')
df = apply_rule(df, "RULE 000", ['',''], '')
df = apply_rule(df, "RULE 000", ['',''], '')
df = apply_rule(df, "RULE 000", ['',''], '')
df = apply_rule(df, "RULE 000", ['',''], '')
df = apply_rule(df, "RULE 000", ['',''], '')
df = apply_rule(df, "RULE 000", ['',''], '')
df = apply_rule(df, "RULE 000", ['',''], '')
df = apply_rule(df, "RULE 000", ['',''], '')


#df = apply_rule(df, "RULE 000", ['',''], '')

print(len(df))
df.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df = df

# Write recipe outputs
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS")
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS.write_with_schema(ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df)
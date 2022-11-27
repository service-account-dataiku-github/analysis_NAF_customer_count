# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NAFCUSTOMER_C360_ACCOUNTS = dataiku.Dataset("NAFCUSTOMER_C360_ACCOUNTS")
NAFCUSTOMER_C360_ACCOUNTS_df = NAFCUSTOMER_C360_ACCOUNTS.get_dataframe()

ACCOUNT_BUNDLER_LIST = dataiku.Dataset("ACCOUNT_BUNDLER_LIST")
ACCOUNT_BUNDLER_LIST_df = ACCOUNT_BUNDLER_LIST.get_dataframe()

ACCOUNT_BUNDLER_LIST_df['IS_BUNDLER'] = True
ACCOUNT_BUNDLER_LIST_df = ACCOUNT_BUNDLER_LIST_df[['EDW_CUSTOMER_NAME','IS_BUNDLER']]

df = pd.merge(NAFCUSTOMER_C360_ACCOUNTS_df,ACCOUNT_BUNDLER_LIST_df, how='left', on='EDW_CUSTOMER_NAME')
df['DUNS'] = df['DUNS'].astype('Int64', errors='ignore')

df.loc[df["IS_BUNDLER"].isnull(),'IS_BUNDLER'] = False
df.loc[df["IS_BUNDLER"],'EDW_CUSTOMER_NAME'] = np.nan

# override --> need to figure out how to deal with these additions
df.loc[df["EDW_CUSTOMER_NAME"]=='EXXONMOBIL PL CONVERSION L1','IS_BUNDLER'] = True

ACCOUNTS_WITH_BUNDLER_df = df

# Write recipe outputs
ACCOUNTS_WITH_BUNDLER = dataiku.Dataset("ACCOUNTS_WITH_BUNDLER")
ACCOUNTS_WITH_BUNDLER.write_with_schema(ACCOUNTS_WITH_BUNDLER_df)
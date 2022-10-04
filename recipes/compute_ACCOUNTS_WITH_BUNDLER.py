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

ACCOUNT_BUNDLER_LIST_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(NAFCUSTOMER_C360_ACCOUNTS_df))
df = pd.merge(NAFCUSTOMER_C360_ACCOUNTS_df,ACCOUNT_BUNDLER_LIST_df, how='left', on='EDW_CUSTOMER_NAME')
print(len(df))
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

ACCOUNTS_WITH_BUNDLER_df = df

# Write recipe outputs
ACCOUNTS_WITH_BUNDLER = dataiku.Dataset("ACCOUNTS_WITH_BUNDLER")
ACCOUNTS_WITH_BUNDLER.write_with_schema(ACCOUNTS_WITH_BUNDLER_df)
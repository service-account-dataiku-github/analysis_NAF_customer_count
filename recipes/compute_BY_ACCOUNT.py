# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS")
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df = ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS.get_dataframe()

CALCULATED_DRAW_UPS = dataiku.Dataset("CALCULATED_DRAW_UPS")
CALCULATED_DRAW_UPS_df = CALCULATED_DRAW_UPS.get_dataframe()

CALCULATED_DRAW_DOWNS = dataiku.Dataset("CALCULATED_DRAW_DOWNS")
CALCULATED_DRAW_DOWNS_df = CALCULATED_DRAW_DOWNS.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
CALCULATED_DRAW_UPS_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_a = ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df[['CUSTOMER_ACCOUNT_ID','CUSTOMER']]
print(len(df_a), "total accounts")

print()

print(len(CALCULATED_DRAW_UPS_df), "with draw ups")
df_down = pd.merge(df_a, CALCULATED_DRAW_UPS_df, on='CUSTOMER_ACCOUNT_ID', how='inner')
print(len(df_down), "accounts joined with draw ups")

print()

print(len(CALCULATED_DRAW_DOWNS_df), "with draw downs")
df_up = pd.merge(df_a, CALCULATED_DRAW_DOWNS_df, on='CUSTOMER_ACCOUNT_ID', how='inner')
print(len(df_up), "accounts joined with draw downs")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_down.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

#BY_ACCOUNT_df = ... # Compute a Pandas dataframe to write into BY_ACCOUNT


# Write recipe outputs
#BY_ACCOUNT = dataiku.Dataset("BY_ACCOUNT")
#BY_ACCOUNT.write_with_schema(BY_ACCOUNT_df)
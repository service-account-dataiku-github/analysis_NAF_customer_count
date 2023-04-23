# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
WEX_and_Non_Managed_Sold_20190101_20230130 = dataiku.Dataset("WEX_and_Non_Managed_Sold_20190101_20230130")
WEX_and_Non_Managed_Sold_20190101_20230130_df = WEX_and_Non_Managed_Sold_20190101_20230130.get_dataframe()
print(len(WEX_and_Non_Managed_Sold_20190101_20230130_df))

Managed_Sold_20190101_20230130 = dataiku.Dataset("Managed_Sold_20190101_20230130")
Managed_Sold_20190101_20230130_df = Managed_Sold_20190101_20230130.get_dataframe()
print(len(Managed_Sold_20190101_20230130_df))

Sold_20230101_20230331 = dataiku.Dataset("Sold_20230101_20230331")
Sold_20230101_20230331_df = Sold_20230101_20230331.get_dataframe()
print(len(Sold_20230101_20230331_df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
WEX_and_Non_Managed_Sold_20190101_20230130_df['DATA_SOURCE'] = 'WEX_and_Non_Managed_Sold'
Managed_Sold_20190101_20230130_df['DATA_SOURCE'] = 'Managed_Sold'

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(WEX_and_Non_Managed_Sold_20190101_20230130_df))
print(len(Managed_Sold_20190101_20230130_df))
df = pd.concat([WEX_and_Non_Managed_Sold_20190101_20230130_df,Managed_Sold_20190101_20230130_df])
print(len(df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df))
df.drop_duplicates(subset=['SOURCE_ACCOUNT_ID'], inplace=True)
print(len(df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

ACCOUNT_NEW_SALES_FULL_df = df

# Write recipe outputs
ACCOUNT_NEW_SALES_FULL = dataiku.Dataset("ACCOUNT_NEW_SALES_FULL")
ACCOUNT_NEW_SALES_FULL.write_with_schema(ACCOUNT_NEW_SALES_FULL_df)
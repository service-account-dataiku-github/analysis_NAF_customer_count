# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED = dataiku.Dataset("NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED")
NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df = NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED.get_dataframe()

print(len(NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df), "rows")

print(len(NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df.CUSTOMER_ACCOUNT_ID.unique()), "accounts")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_v = NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df

print(len(df_v))
df_v.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

#CALCULATED_DRAW_UPS_df = NAFCUSTOMER_REVENUE_BY_CUSTOMER_LIMITED_df # For this sample code, simply copy input to output

# Write recipe outputs
#CALCULATED_DRAW_UPS = dataiku.Dataset("CALCULATED_DRAW_UPS")
#CALCULATED_DRAW_UPS.write_with_schema(CALCULATED_DRAW_UPS_df)
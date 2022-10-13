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


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

BY_ACCOUNT_df = ... # Compute a Pandas dataframe to write into BY_ACCOUNT


# Write recipe outputs
BY_ACCOUNT = dataiku.Dataset("BY_ACCOUNT")
BY_ACCOUNT.write_with_schema(BY_ACCOUNT_df)

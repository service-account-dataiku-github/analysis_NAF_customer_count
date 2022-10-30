# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
AccountScorecard_10_28 = dataiku.Dataset("AccountScorecard_10_28")
AccountScorecard_10_28_df = AccountScorecard_10_28.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

ACCOUNT_NEW_SALES_df = AccountScorecard_10_28_df # For this sample code, simply copy input to output


# Write recipe outputs
ACCOUNT_NEW_SALES = dataiku.Dataset("ACCOUNT_NEW_SALES")
ACCOUNT_NEW_SALES.write_with_schema(ACCOUNT_NEW_SALES_df)

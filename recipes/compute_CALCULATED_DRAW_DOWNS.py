# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NAFCUSTOMER_C360_ACCOUNTS = dataiku.Dataset("NAFCUSTOMER_C360_ACCOUNTS")
NAFCUSTOMER_C360_ACCOUNTS_df = NAFCUSTOMER_C360_ACCOUNTS.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

CALCULATED_DRAW_DOWNS_df = NAFCUSTOMER_C360_ACCOUNTS_df # For this sample code, simply copy input to output


# Write recipe outputs
CALCULATED_DRAW_DOWNS = dataiku.Dataset("CALCULATED_DRAW_DOWNS")
CALCULATED_DRAW_DOWNS.write_with_schema(CALCULATED_DRAW_DOWNS_df)

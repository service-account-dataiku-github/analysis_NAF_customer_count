# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER = dataiku.Dataset("NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER")
NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER_df = NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER.get_dataframe()

NAFCUSTOMER_RDW_CONVERSIONS = dataiku.Dataset("NAFCUSTOMER_RDW_CONVERSIONS")
NAFCUSTOMER_RDW_CONVERSIONS_df = NAFCUSTOMER_RDW_CONVERSIONS.get_dataframe()

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(NAFCUSTOMER_LONG_HISTORY_MATCHES_df))
print(len(NAFCUSTOMER_RDW_CONVERSIONS_df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Write recipe outputs
NAFCUSTOMER_LONG_HISTORY_MATCHES = dataiku.Dataset("NAFCUSTOMER_LONG_HISTORY_MATCHES")
NAFCUSTOMER_LONG_HISTORY_MATCHES.write_with_schema(NAFCUSTOMER_LONG_HISTORY_MATCHES_df)
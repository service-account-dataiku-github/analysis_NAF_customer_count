# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NAFCUSTOMER_C360_ACCOUNTS = dataiku.Dataset("NAFCUSTOMER_C360_ACCOUNTS")
NAFCUSTOMER_C360_ACCOUNTS_df = NAFCUSTOMER_C360_ACCOUNTS.get_dataframe()
ACCOUNT_BUNDLER_LIST = dataiku.Dataset("ACCOUNT_BUNDLER_LIST")
ACCOUNT_BUNDLER_LIST_df = ACCOUNT_BUNDLER_LIST.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

ACCOUNTS_WITH_BUNDLER_df = ... # Compute a Pandas dataframe to write into ACCOUNTS_WITH_BUNDLER


# Write recipe outputs
ACCOUNTS_WITH_BUNDLER = dataiku.Dataset("ACCOUNTS_WITH_BUNDLER")
ACCOUNTS_WITH_BUNDLER.write_with_schema(ACCOUNTS_WITH_BUNDLER_df)

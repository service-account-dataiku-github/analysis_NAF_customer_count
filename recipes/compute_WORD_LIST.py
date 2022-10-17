# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
BY_ACCOUNT = dataiku.Dataset("BY_ACCOUNT")
BY_ACCOUNT_df = BY_ACCOUNT.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

WORD_LIST_df = BY_ACCOUNT_df # For this sample code, simply copy input to output


# Write recipe outputs
WORD_LIST = dataiku.Dataset("WORD_LIST")
WORD_LIST.write_with_schema(WORD_LIST_df)

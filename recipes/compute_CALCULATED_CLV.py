# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NAFCUSTOMER_NEW_ACCOUNTS_IN_2021 = dataiku.Dataset("NAFCUSTOMER_NEW_ACCOUNTS_IN_2021")
NAFCUSTOMER_NEW_ACCOUNTS_IN_2021_df = NAFCUSTOMER_NEW_ACCOUNTS_IN_2021.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

CALCULATED_CLV_df = NAFCUSTOMER_NEW_ACCOUNTS_IN_2021_df # For this sample code, simply copy input to output


# Write recipe outputs
CALCULATED_CLV = dataiku.Dataset("CALCULATED_CLV")
CALCULATED_CLV.write_with_schema(CALCULATED_CLV_df)

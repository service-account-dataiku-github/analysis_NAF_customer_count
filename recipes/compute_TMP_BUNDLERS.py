# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NAFCUSTOMER_C360_ACCOUNTS = dataiku.Dataset("NAFCUSTOMER_C360_ACCOUNTS")
NAFCUSTOMER_C360_ACCOUNTS_df = NAFCUSTOMER_C360_ACCOUNTS.get_dataframe()
EV_Customer_Segmentation = dataiku.Dataset("EV_Customer_Segmentation")
EV_Customer_Segmentation_df = EV_Customer_Segmentation.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

TMP_BUNDLERS_df = ... # Compute a Pandas dataframe to write into TMP_BUNDLERS


# Write recipe outputs
TMP_BUNDLERS = dataiku.Dataset("TMP_BUNDLERS")
TMP_BUNDLERS.write_with_schema(TMP_BUNDLERS_df)

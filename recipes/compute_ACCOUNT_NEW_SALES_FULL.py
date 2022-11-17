# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
WEX_and_Non_Managed_Sold_20190101_20220920 = dataiku.Dataset("WEX_and_Non_Managed_Sold_20190101_20220920")
WEX_and_Non_Managed_Sold_20190101_20220920_df = WEX_and_Non_Managed_Sold_20190101_20220920.get_dataframe()
Managed_Sold_20190101_20220920 = dataiku.Dataset("Managed_Sold_20190101_20220920")
Managed_Sold_20190101_20220920_df = Managed_Sold_20190101_20220920.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

ACCOUNT_NEW_SALES_FULL_df = ... # Compute a Pandas dataframe to write into ACCOUNT_NEW_SALES_FULL


# Write recipe outputs
ACCOUNT_NEW_SALES_FULL = dataiku.Dataset("ACCOUNT_NEW_SALES_FULL")
ACCOUNT_NEW_SALES_FULL.write_with_schema(ACCOUNT_NEW_SALES_FULL_df)

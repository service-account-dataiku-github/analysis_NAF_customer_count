# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
Account_Party_extract = dataiku.Dataset("Account_Party_extract")
Account_Party_extract_df = Account_Party_extract.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

MDM_ACCOUNT_PARTY_EXTRACT_df = Account_Party_extract_df # For this sample code, simply copy input to output


# Write recipe outputs
MDM_ACCOUNT_PARTY_EXTRACT = dataiku.Dataset("MDM_ACCOUNT_PARTY_EXTRACT")
MDM_ACCOUNT_PARTY_EXTRACT.write_with_schema(MDM_ACCOUNT_PARTY_EXTRACT_df)

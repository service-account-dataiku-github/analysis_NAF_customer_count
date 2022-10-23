# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING = dataiku.Dataset("MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING")
MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING_df = MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING.get_dataframe()

MATCHES_1_TO_1_STAGING = dataiku.Dataset("MATCHES_1_TO_1_STAGING")
MATCHES_1_TO_1_STAGING_df = MATCHES_1_TO_1_STAGING.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_1_N = MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING_df
print(len(df_1_N))
df_1_N.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_1_1 = MATCHES_1_TO_1_STAGING_df
print(len(df_1_1))
df_1_1.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from difflib import SequenceMatcher
import Levenshtein

idx = 0
exception_count = 0

_verified = []
_verified_match = []

_exception = []
_exception_match = []

for index, row in df_1_1.iterrows():

    idx+=1
    
    customer = row['CUSTOMER']
    match_customer = row['MATCH_CUSTOMER']
    
    r = Levenshtein.ratio(customer, match_customer)
    if r<0.8:
        _exception.append(customer)
        _exception_match.append(match_customer)
    else:
        _verified.append(customer)
        _verified_match.append(match_customer)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
_CUSTOMER = []
_CUSTOMER_CLC = []

for i in range(0, len(_verified)):
    
    _CUSTOMER.append(_verified[i])
    _CUSTOMER.append(_verified_match[i])
    _CUSTOMER_CLC.append(_verified[i])
    _CUSTOMER_CLC.append(_verified[i])
    
df_clc = pd.DataFrame(_CUSTOMER, columns=['CUSTOMER'])
df_clc['CUSTOMER_CLC'] = _CUSTOMER_CLC
print(len(df_clc))
df_clc.tail(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

MATCHES_VERIFIED_df = df_clc

# Write recipe outputs
MATCHES_VERIFIED = dataiku.Dataset("MATCHES_VERIFIED")
MATCHES_VERIFIED.write_with_schema(MATCHES_VERIFIED_df)
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from difflib import SequenceMatcher
import Levenshtein

# Read recipe inputs
MATCHES_1_TO_1_QUEUED = dataiku.Dataset("MATCHES_1_TO_1_QUEUED")
MATCHES_1_TO_1_QUEUED_df = MATCHES_1_TO_1_QUEUED.get_dataframe()

MATCHES_1_TO_N_QUEUED = dataiku.Dataset("MATCHES_1_TO_N_QUEUED")
MATCHES_1_TO_N_QUEUED_df = MATCHES_1_TO_N_QUEUED.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_1_N = MATCHES_1_TO_1_QUEUED_df
df_1_N['distance'] = df_1_N.apply(lambda x: Levenshtein.ratio(x['CUSTOMER'],x['MATCH_CUSTOMER']),axis=1)
print(len(df_1_N))
df_1_N = df_1_N[df_1_N.distance>0.8]
print(len(df_1_N))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

VERIFIED_MATCHES_df = df_1_N

# Write recipe outputs
VERIFIED_MATCHES = dataiku.Dataset("VERIFIED_MATCHES")
VERIFIED_MATCHES.write_with_schema(VERIFIED_MATCHES_df)
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
MATCHES_1_TO_N_STAGING_V = dataiku.Dataset("MATCHES_1_TO_N_STAGING_V")
MATCHES_1_TO_N_STAGING_V_df = MATCHES_1_TO_N_STAGING_V.get_dataframe()
MATCHES_1_TO_1_STAGING_V = dataiku.Dataset("MATCHES_1_TO_1_STAGING_V")
MATCHES_1_TO_1_STAGING_V_df = MATCHES_1_TO_1_STAGING_V.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

MATCHES_VERIFIED_V_df = ... # Compute a Pandas dataframe to write into MATCHES_VERIFIED_V


# Write recipe outputs
MATCHES_VERIFIED_V = dataiku.Dataset("MATCHES_VERIFIED_V")
MATCHES_VERIFIED_V.write_with_schema(MATCHES_VERIFIED_V_df)

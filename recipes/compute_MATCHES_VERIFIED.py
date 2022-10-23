# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING = dataiku.Dataset("MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING")
MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING_df = MATCHES_1_TO_N_FOR_MANUAL_REVIEW_STAGING.get_dataframe()
MATCHES_1_TO_1_STAGING = dataiku.Dataset("MATCHES_1_TO_1_STAGING")
MATCHES_1_TO_1_STAGING_df = MATCHES_1_TO_1_STAGING.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

MATCHES_VERIFIED_df = ... # Compute a Pandas dataframe to write into MATCHES_VERIFIED


# Write recipe outputs
MATCHES_VERIFIED = dataiku.Dataset("MATCHES_VERIFIED")
MATCHES_VERIFIED.write_with_schema(MATCHES_VERIFIED_df)

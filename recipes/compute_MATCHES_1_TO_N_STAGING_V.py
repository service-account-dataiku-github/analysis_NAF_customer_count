# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
CALCULATED_DRAW_DOWNS = dataiku.Dataset("CALCULATED_DRAW_DOWNS")
CALCULATED_DRAW_DOWNS_df = CALCULATED_DRAW_DOWNS.get_dataframe()
CALCULATED_DRAW_UPS = dataiku.Dataset("CALCULATED_DRAW_UPS")
CALCULATED_DRAW_UPS_df = CALCULATED_DRAW_UPS.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

MATCHES_1_TO_N_STAGING_V_df = ... # Compute a Pandas dataframe to write into MATCHES_1_TO_N_STAGING_V
MATCHES_1_TO_1_STAGING_V_df = ... # Compute a Pandas dataframe to write into MATCHES_1_TO_1_STAGING_V


# Write recipe outputs
MATCHES_1_TO_N_STAGING_V = dataiku.Dataset("MATCHES_1_TO_N_STAGING_V")
MATCHES_1_TO_N_STAGING_V.write_with_schema(MATCHES_1_TO_N_STAGING_V_df)
MATCHES_1_TO_1_STAGING_V = dataiku.Dataset("MATCHES_1_TO_1_STAGING_V")
MATCHES_1_TO_1_STAGING_V.write_with_schema(MATCHES_1_TO_1_STAGING_V_df)

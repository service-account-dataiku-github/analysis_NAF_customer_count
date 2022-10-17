# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
MATCHES_1_TO_N_FOR_MANUAL_REVIEW = dataiku.Dataset("MATCHES_1_TO_N_FOR_MANUAL_REVIEW")
MATCHES_1_TO_N_FOR_MANUAL_REVIEW_df = MATCHES_1_TO_N_FOR_MANUAL_REVIEW.get_dataframe()

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

MATCHES_1_TO_N_BEST_MATCHES_df = MATCHES_1_TO_N_FOR_MANUAL_REVIEW_df # For this sample code, simply copy input to output


# Write recipe outputs
MATCHES_1_TO_N_BEST_MATCHES = dataiku.Dataset("MATCHES_1_TO_N_BEST_MATCHES")
MATCHES_1_TO_N_BEST_MATCHES.write_with_schema(MATCHES_1_TO_N_BEST_MATCHES_df)

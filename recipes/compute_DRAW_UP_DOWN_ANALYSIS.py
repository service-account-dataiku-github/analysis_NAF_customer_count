# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
CALCULATED_CARD_DRAW_UPS_FULL = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS_FULL")
CALCULATED_CARD_DRAW_UPS_FULL_df = CALCULATED_CARD_DRAW_UPS_FULL.get_dataframe()
CALCULATED_CARD_DRAW_DOWNS_FULL = dataiku.Dataset("CALCULATED_CARD_DRAW_DOWNS_FULL")
CALCULATED_CARD_DRAW_DOWNS_FULL_df = CALCULATED_CARD_DRAW_DOWNS_FULL.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

DRAW_UP_DOWN_ANALYSIS_df = ... # Compute a Pandas dataframe to write into DRAW_UP_DOWN_ANALYSIS


# Write recipe outputs
DRAW_UP_DOWN_ANALYSIS = dataiku.Dataset("DRAW_UP_DOWN_ANALYSIS")
DRAW_UP_DOWN_ANALYSIS.write_with_schema(DRAW_UP_DOWN_ANALYSIS_df)

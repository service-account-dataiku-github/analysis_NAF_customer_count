# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
CALCULATED_CARD_DRAW_UPS = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS")
CALCULATED_CARD_DRAW_UPS_df = CALCULATED_CARD_DRAW_UPS.get_dataframe()
CALCULATED_CARD_DRAW_DOWNS = dataiku.Dataset("CALCULATED_CARD_DRAW_DOWNS")
CALCULATED_CARD_DRAW_DOWNS_df = CALCULATED_CARD_DRAW_DOWNS.get_dataframe()


# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

CONSOLIDATED_MATCHES_df = ... # Compute a Pandas dataframe to write into CONSOLIDATED_MATCHES


# Write recipe outputs
CONSOLIDATED_MATCHES = dataiku.Dataset("CONSOLIDATED_MATCHES")
CONSOLIDATED_MATCHES.write_with_schema(CONSOLIDATED_MATCHES_df)

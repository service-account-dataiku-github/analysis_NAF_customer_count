# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NAFCUSTOMER_ACTIVE_CARDS_FULL = dataiku.Dataset("NAFCUSTOMER_ACTIVE_CARDS_FULL")
NAFCUSTOMER_ACTIVE_CARDS_FULL_df = NAFCUSTOMER_ACTIVE_CARDS_FULL.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

CALCULATED_CARD_DRAW_UPS_FULL_df = NAFCUSTOMER_ACTIVE_CARDS_FULL_df # For this sample code, simply copy input to output


# Write recipe outputs
CALCULATED_CARD_DRAW_UPS_FULL = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS_FULL")
CALCULATED_CARD_DRAW_UPS_FULL.write_with_schema(CALCULATED_CARD_DRAW_UPS_FULL_df)

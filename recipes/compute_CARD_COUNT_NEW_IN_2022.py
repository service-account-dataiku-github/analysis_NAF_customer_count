# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
NAFCUSTOMER_NEW_CUSTOMERS_IN_2022_WITH_SALES_FLAG_AND_ACTIVE_CARDS = dataiku.Dataset("NAFCUSTOMER_NEW_CUSTOMERS_IN_2022_WITH_SALES_FLAG_AND_ACTIVE_CARDS")
NAFCUSTOMER_NEW_CUSTOMERS_IN_2022_WITH_SALES_FLAG_AND_ACTIVE_CARDS_df = NAFCUSTOMER_NEW_CUSTOMERS_IN_2022_WITH_SALES_FLAG_AND_ACTIVE_CARDS.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

CARD_COUNT_NEW_IN_2022_df = NAFCUSTOMER_NEW_CUSTOMERS_IN_2022_WITH_SALES_FLAG_AND_ACTIVE_CARDS_df # For this sample code, simply copy input to output


# Write recipe outputs
CARD_COUNT_NEW_IN_2022 = dataiku.Dataset("CARD_COUNT_NEW_IN_2022")
CARD_COUNT_NEW_IN_2022.write_with_schema(CARD_COUNT_NEW_IN_2022_df)

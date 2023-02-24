# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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
print(len(CARD_COUNT_NEW_IN_2022_df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = CARD_COUNT_NEW_IN_2022_df.copy()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_customers_with_card_count = df.groupby(['CUSTOMER_ID','CUSTOMER']).ACTIVE_CARD_COUNT.sum().reset_index()

df_customers_with_card_count['FLEET_SIZE'] = 'NOT SET'
df_customers_with_card_count.loc[df_customers_with_card_count.ACTIVE_CARD_COUNT.between(0,5), 'FLEET_SIZE'] = '(<=5 cards)'
df_customers_with_card_count.loc[df_customers_with_card_count.ACTIVE_CARD_COUNT.between(6,100), 'FLEET_SIZE'] = '(between 6 and 100 cards)'
df_customers_with_card_count.loc[df_customers_with_card_count.ACTIVE_CARD_COUNT>100, 'FLEET_SIZE'] = '(>100 cards)'
df_customers_with_card_count.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_customers_with_card_count[df_customers_with_card_count.FLEET_SIZE=='(>100 cards)'].head(100)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_customers_with_card_count.groupby('FLEET_SIZE').ACTIVE_CARD_COUNT.sum()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_customers_with_card_count.groupby('FLEET_SIZE').CUSTOMER_ID.nunique()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
CARD_COUNT_NEW_IN_2022 = dataiku.Dataset("CARD_COUNT_NEW_IN_2022")
CARD_COUNT_NEW_IN_2022.write_with_schema(CARD_COUNT_NEW_IN_2022_df)
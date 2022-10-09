# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS")
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df = ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS.get_dataframe()

NAFCUSTOMER_CUSTOMERS_WITH_MORE_THAN_X_REVENUE = dataiku.Dataset("NAFCUSTOMER_CUSTOMERS_WITH_MORE_THAN_X_REVENUE")
NAFCUSTOMER_CUSTOMERS_WITH_MORE_THAN_X_REVENUE_df = NAFCUSTOMER_CUSTOMERS_WITH_MORE_THAN_X_REVENUE.get_dataframe()

# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = NAFCUSTOMER_CUSTOMERS_WITH_MORE_THAN_X_REVENUE_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# find all customer_names that have other customer names that start with this customer name

idx = 0
unique_customer_names = df['CUSTOMER'].unique()
full_set_n = len(unique_customer_names)

list_customers_ = []
list_potential_matches_ = []

report_every_n = 1000
save_every_n = 10000

for n in unique_customer_names:
    idx+=1
    df_f = df[(df['CUSTOMER'].str.startswith(n, na=False))&(df['CUSTOMER']!=n)]
    match_list = df_f['CUSTOMER'].unique()
    if len(match_list)>0:
        list_customers_.append(n)
        list_potential_matches_.append(match_list)

    if (idx % report_every_n == 0):
        print(idx, "iterations", len(list_customers_), "with potential matches", full_set_n-idx, "remaining", round(idx/full_set_n,2), "% complete")
        
    if (idx % save_every_n == 0):
        print('SAVING DATAFRAME')        
        df_candidates = pd.DataFrame(list_customers_)
        df_candidates.columns = ['CUSTOMER']
        df_candidates['POTENTIAL_MATCHES'] = list_potential_matches_

        ACCOUNTS_WITH_POTENTIAL_DUPLICATES_df = df_candidates

        # Write recipe outputs
        ACCOUNTS_WITH_POTENTIAL_DUPLICATES = dataiku.Dataset("ACCOUNTS_WITH_POTENTIAL_DUPLICATES")
        ACCOUNTS_WITH_POTENTIAL_DUPLICATES.write_with_schema(ACCOUNTS_WITH_POTENTIAL_DUPLICATES_df)
        print(len(ACCOUNTS_WITH_POTENTIAL_DUPLICATES_df), "written")
        print("continuing on with checks....")
        
print(len(list_customers_))
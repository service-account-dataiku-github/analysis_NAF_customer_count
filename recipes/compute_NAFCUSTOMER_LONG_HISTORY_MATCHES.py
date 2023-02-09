# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

ACCOUNT_BUNDLER_LIST = dataiku.Dataset("ACCOUNT_BUNDLER_LIST")
ACCOUNT_BUNDLER_LIST_df = ACCOUNT_BUNDLER_LIST.get_dataframe()
print("Account Bundlers:", len(ACCOUNT_BUNDLER_LIST_df))

#NAFCUSTOMER_RDW_CONVERSIONS = dataiku.Dataset("NAFCUSTOMER_RDW_CONVERSIONS")
#NAFCUSTOMER_RDW_CONVERSIONS_df = NAFCUSTOMER_RDW_CONVERSIONS.get_dataframe()
#print(len(NAFCUSTOMER_RDW_CONVERSIONS_df))

# Read recipe inputs
NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER = dataiku.Dataset("NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER")
NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER_df = NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER.get_dataframe()
print(len(NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER_df))

print(NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER_df.YEAR_NUMBER.min())

print("Min Year:", NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER_df.YEAR_NUMBER.min())
print("Max Year:", NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER_df.YEAR_NUMBER.max())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_a = NAFCUSTOMER_ACCOUNT_ACTIVE_CARDS_BY_QUARTER_df

df_a.columns = ['CUSTOMER_ACCOUNT_ID','CUSTOMER_ACCOUNT_NAME','EDW_CUSTOMER_NAME','ACCOUNT_SINCE_DATE','CUSTOMER_SOURCE_SYSTEM_CODE','ACCOUNT_OPEN_DATE','ACCOUNT_CLOSED_DATE','ATTRITION_TYPE_NAME','ATTRITION_REASON_CODE','ATTRITION_REASON_DESC','YEAR_NUMBER','QUARTER_NUMBER','ACTIVE_CARD_COUNT']

df_a['CUSTOMER_ACCOUNT_ID'] = df_a['CUSTOMER_ACCOUNT_ID'].astype('Int64', errors='ignore')
df_a['CUSTOMER_ACCOUNT_NAME'] = df_a['CUSTOMER_ACCOUNT_NAME'].str.upper()
df_a['EDW_CUSTOMER_NAME'] = df_a['EDW_CUSTOMER_NAME'].str.upper()

ACCOUNT_BUNDLER_LIST_df['IS_BUNDLER'] = True
ACCOUNT_BUNDLER_LIST_df = ACCOUNT_BUNDLER_LIST_df[['EDW_CUSTOMER_NAME','IS_BUNDLER']]
ACCOUNT_BUNDLER_LIST_df.head()

df = pd.merge(df_a,ACCOUNT_BUNDLER_LIST_df, how='left', on='EDW_CUSTOMER_NAME')
df.loc[df["IS_BUNDLER"].isnull(),'IS_BUNDLER'] = False
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['CUSTOMER'] = np.nan
df['CUST_CALC_SOURCE'] = 'Unknown'
df.loc[df.IS_BUNDLER,'EDW_CUSTOMER_NAME'] = np.nan

df.loc[~df['EDW_CUSTOMER_NAME'].isnull(),'CUSTOMER'] = df["EDW_CUSTOMER_NAME"]
df.loc[~df['EDW_CUSTOMER_NAME'].isnull(),'CUST_CALC_SOURCE'] = 'EDW'

df.loc[df['CUSTOMER'].isnull(),'CUST_CALC_SOURCE'] = 'ACCOUNT'
df.loc[df['CUSTOMER'].isnull(),'CUSTOMER'] = df.CUSTOMER_ACCOUNT_NAME


ending_tokens = [' 2', ' 3', ' 4', ' 04', ' 5', ' 6', ' 7', ' 8', ' 9',' (2)',
                 ' (3)',' (04)',' (4)', ' (5)', ' (6)', ' (7)', ' (8)',
                 ' (9)',' (25)','  (32)', ' AD', ' LD', 'L1']

df['CUSTOMER'].str.strip()

for s in ending_tokens:
    index_offset = -1*(len(s))
    df.loc[df['CUSTOMER'].str.endswith(s, na=False),"CUSTOMER"] = df['CUSTOMER'].str[:index_offset]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Convert YEAR AND QUARTER INTO DATE REPRESENTING THE FIRST DAY OF THE QUARTER
df['REVENUE_DATE'] = ((3*df.QUARTER_NUMBER)-2).astype(str) + "/1/" + (df.YEAR_NUMBER).astype(str)
df['REVENUE_DATE'] = pd.to_datetime(df["REVENUE_DATE"])
df['ACCOUNT_SINCE_DATE'] = pd.to_datetime(df['ACCOUNT_SINCE_DATE'])

#df['CUSTOMER STATE']
#df.loc[df['CUSTOMER_ACCOUNT_NAME'].str.endswith(s, na=False),"CUSTOMER_ACCOUNT_NAME"] = df['CUSTOMER_ACCOUNT_NAME'].str[:index_offset]

df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.CUST_CALC_SOURCE.value_counts(dropna=False)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_cust_since = df.groupby(['CUSTOMER']).ACCOUNT_SINCE_DATE.min().reset_index()
df_cust_since = df_cust_since.sort_values(by=['CUSTOMER'], ascending=True)
df_cust_since.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_cust_max_revenue = df.groupby(['CUSTOMER']).REVENUE_DATE.max().reset_index()
df_cust_max_revenue.columns = ['CUSTOMER','MAX_REVENUE_DATE']
df_cust_max_revenue = df_cust_max_revenue.sort_values(by=['CUSTOMER'], ascending=True)
df_cust_max_revenue.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_cust_min_revenue = df.groupby(['CUSTOMER']).REVENUE_DATE.min().reset_index()
df_cust_min_revenue.columns = ['CUSTOMER','MIN_REVENUE_DATE']
df_cust_min_revenue = df_cust_min_revenue.sort_values(by=['CUSTOMER'], ascending=True)
df_cust_min_revenue.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Write recipe outputs
NAFCUST_HISTORY_TENURE_2010_DORMANT = dataiku.Dataset("NAFCUST_HISTORY_TENURE_2010_DORMANT")
NAFCUST_HISTORY_TENURE_2010_DORMANT.write_with_schema(NAFCUSTOMER_LONG_HISTORY_MATCHES_df)

NAFCUST_HISTORY_TENURE_2010_2011_ORIGINATE = dataiku.Dataset("NAFCUST_HISTORY_TENURE_2010_2011_ORIGINATE")
NAFCUST_HISTORY_TENURE_2010_2011_ORIGINATE.write_with_schema(NAFCUST_HISTORY_TENURE_2010_2011_ORIGINATE_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

#df_account_with_customer = df[['CUSTOMER_ACCOUNT_ID','CUSTOMER']].copy()
#df_account_with_customer = df_account_with_customer.drop_duplicates(subset='CUSTOMER_ACCOUNT_ID')
#print(len(df_account_with_customer))
#df_account_with_customer.head()

#df_g.columns = ['CUSTOMER','ACCOUNT_COUNT']
#df_g = df_g.sort_values(by=['ACCOUNT_COUNT'], ascending=False)
#df_g.head(100)

#NAFCUSTOMER_RDW_CONVERSIONS_df.head()
#NAFCUSTOMER_RDW_CONVERSIONS_df.STATUS_DATE.min()

#NAFCUSTOMER_RDW_CONVERSIONS_df.head()
#df_conv = NAFCUSTOMER_RDW_CONVERSIONS_df[['FLEET_ID','CLASSIC_ACCOUNT_NUMBER','FLEET_NAME']].copy()
#print(len(df_conv))
#df_conv = df_conv[~df_conv.CLASSIC_ACCOUNT_NUMBER.isnull()]
#print(len(df_conv))

#df_conv.columns = ['CUSTOMER_ACCOUNT_ID', 'CLASSIC_CUSTOMER_ACCOUNT_ID', 'FLEET_NAME']
#df_conv['CUSTOMER_ACCOUNT_ID'] = df_conv['CUSTOMER_ACCOUNT_ID'].astype('Int64', errors='ignore')
#df_conv['CLASSIC_CUSTOMER_ACCOUNT_ID'] = pd.to_numeric(df_conv['CLASSIC_CUSTOMER_ACCOUNT_ID'], errors='coerce')
#df_conv = df_conv[~df_conv.CLASSIC_CUSTOMER_ACCOUNT_ID.isnull()]
#df_conv['CLASSIC_CUSTOMER_ACCOUNT_ID'] = df_conv['CLASSIC_CUSTOMER_ACCOUNT_ID'].astype('int64', errors='ignore')
#df_conv['CLASSIC_CUSTOMER_ACCOUNT_ID'] = df_conv['CLASSIC_CUSTOMER_ACCOUNT_ID'].astype('Int64', errors='ignore')

#df_conv = pd.merge(df_conv, df_account_with_customer, on='CUSTOMER_ACCOUNT_ID', how='inner')
#print(len(df_conv))
#df_conv = df_conv[['CLASSIC_CUSTOMER_ACCOUNT_ID','CUSTOMER']]
#df_conv.columns = ['CLASSIC_CUSTOMER_ACCOUNT_ID','CUSTOMER_CONVERTED_TO']
#df_conv.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df = df[['CUSTOMER_ACCOUNT_ID','CUSTOMER_ACCOUNT_NAME','YEAR_NUMBER','QUARTER_NUMBER','ACTIVE_CARD_COUNT','CUSTOMER','CUST_CALC_SOURCE']]
#print(len(df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#print(len(df))
#df = pd.merge(df, df_conv, left_on='CUSTOMER_ACCOUNT_ID', right_on='CLASSIC_CUSTOMER_ACCOUNT_ID', how='left')
#print(len(df))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df.CUSTOMER_CONVERTED_TO.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df.loc[~df['CUSTOMER_CONVERTED_TO'].isnull(),'CUSTOMER'] = df.CUSTOMER_CONVERTED_TO

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

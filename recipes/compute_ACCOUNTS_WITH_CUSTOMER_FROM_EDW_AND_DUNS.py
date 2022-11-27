# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import string

#===============
# Purpose: Calculate NAFleet Customer hierarchy from best known sources of account groupings + custom rules
# sources: EDW, Ebx, RDW Conversions, Drawdown/Drawups + entity matching
#
# Oct-Dec 2022
# Daniel VanderMeer,
# email: daniel.vandermeer@wexinc.com
#==============

# Read recipe inputs

#===============
# Data Set: Accounts with Bundler and Duns
# columns: CUSTOMER_ACCOUNT_ID, CUSTOMER_ACCOUNT_NAME, EDW_CUSTOMER_NAME, DUNS, IS_BUNDLER
# Note Account bundlers are ~275 known instances where EDW customer names do not describe customer entities
# in most cases these are either partner or program names
# examples: CIRCLE K STORES PRIMARY, WEX FLEET UNIVERSAL PRIMARY
# the IS_BUNDLER columns allows the algorithm to ignore these cases

ACCOUNTS_WITH_BUNDLER_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_BUNDLER_AND_DUNS")
ACCOUNTS_WITH_BUNDLER_AND_DUNS_df = ACCOUNTS_WITH_BUNDLER_AND_DUNS.get_dataframe()

NAFCUSTOMER_MDM_ACCOUNT_WITH_BUSINESS_ID = dataiku.Dataset("NAFCUSTOMER_MDM_ACCOUNT_WITH_BUSINESS_ID")
NAFCUSTOMER_MDM_ACCOUNT_WITH_BUSINESS_ID_df = NAFCUSTOMER_MDM_ACCOUNT_WITH_BUSINESS_ID.get_dataframe()

# MDM matches, shared by Wes Corbin during the week of Nov 17, 2022
# key columns: ACCOUNTNUMBER, WEXBUSINESSID, NAME, DUNS
#ACCOUNTS_PARTY_EXTRACT = dataiku.Dataset("Account_Party_extract")
#ACCOUNTS_PARTY_EXTRACT_df = ACCOUNTS_PARTY_EXTRACT.get_dataframe()
#ACCOUNTS_PARTY_EXTRACT_df = ACCOUNTS_PARTY_EXTRACT_df[~ACCOUNTS_PARTY_EXTRACT_df.ACCOUNTNUMBER.str.contains('-', na=False)]
#ACCOUNTS_PARTY_EXTRACT_df['ACCOUNTNUMBER'] = ACCOUNTS_PARTY_EXTRACT_df['ACCOUNTNUMBER'].str.strip()
#ACCOUNTS_PARTY_EXTRACT_df['ACCOUNTNUMBER'] = ACCOUNTS_PARTY_EXTRACT_df['ACCOUNTNUMBER'].astype('float')
#ACCOUNTS_PARTY_EXTRACT_df['ACCOUNTNUMBER'] = ACCOUNTS_PARTY_EXTRACT_df['ACCOUNTNUMBER'].astype('Int64')

# Verified matches, identified by matching algorithm
# matching algorithm combines draw downs and draw ups along with name entity matching
# key columns: CUSTOMER, MATCH_CUSTOMER, DRAW_UP_DATE, distance
MATCHES_VERIFIED = dataiku.Dataset("VERIFIED_MATCHES")
MATCHES_VERIFIED_df = MATCHES_VERIFIED.get_dataframe()

# RDW Conversions
# Conversion teams will track conversions in RDW
# Here we use the source as a way to combine known conversions not yet handled by our matching algorithm or the MDM
RDW_CONVERSIONS = dataiku.Dataset("NAFCUSTOMER_RDW_CONVERSIONS")
RDW_CONVERSIONS_df = RDW_CONVERSIONS.get_dataframe()

# dataset that contains all New Sales 2019-2022 Current
# this dataset comes from Alan Hougham which originates in SAP
# We don't use this dataset in the logic Customer Hierarchy
# It is joined to the detailed dataset (ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df)
# and then we use it as a reconciliation step:
# All new customer accounts in 2019-2022 SHOULD be in this dataset
# new accounts not in this dataset are likely existing accounts that have been converted to a new account
ACCOUNT_NEW_SALES_FULL = dataiku.Dataset("ACCOUNT_NEW_SALES_FULL")
ACCOUNT_NEW_SALES_FULL_df = ACCOUNT_NEW_SALES_FULL.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Matches Verified come from an earlier step in the DataIku Flow
# Accounts are matched on drawdowns, drawups and account name similarity
# key columns: CUSTOMER, MATCH_CUSTOMER, DRAW_UP_DATE, distance
df_matches_verified = MATCHES_VERIFIED_df
df_matches_verified["CUSTOMER"] = df_matches_verified['CUSTOMER'].str.translate(str.maketrans('', '', string.punctuation))
df_matches_verified["MATCH_CUSTOMER"] = df_matches_verified['MATCH_CUSTOMER'].str.translate(str.maketrans('', '', string.punctuation))
df_matches_verified.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import warnings
warnings.filterwarnings(action='once')

df = ACCOUNTS_WITH_BUNDLER_AND_DUNS_df.copy()
print(len(df))

#=====================
# Source: DNB
# DNB levels available (high to low): DNB_GLOBAL_ULT_NAME, DNB_DOMESTIC_ULTIMATE_NAME, DNB_HQ_NAME, DNB_BUSINESS_NAME
# DNB CUSTOMER is the highest available DUNS level customer
# DNB LEVEL keeps track of the DUNS level used
# default DNB CUSTOMER and DNB LEVEL to Null and None

# prep DNB DUNS level columns, cast to upper case, remove punctuation
df['DUNS'] = df['DUNS'].astype('Int64', errors='ignore')
df['DNB_DUNS_NUMBER'] = df['DNB_DUNS_NUMBER'].astype('Int64', errors='ignore')
df['DNB_BUSINESS_NAME'] = df['DNB_BUSINESS_NAME'].str.upper()
df["DNB_BUSINESS_NAME"] = df['DNB_BUSINESS_NAME'].str.translate(str.maketrans('', '', string.punctuation))

# prep DNB DUNS Global Ultimate columns, cast to upper case, remove punctuation
df['DNB_GLOBAL_ULT_NUMBER'] = df['DNB_GLOBAL_ULT_NUMBER'].astype('Int64', errors='ignore')
df['DNB_GLOBAL_ULT_NAME'] = df['DNB_GLOBAL_ULT_NAME'].str.upper()
df["DNB_GLOBAL_ULT_NAME"] = df['DNB_GLOBAL_ULT_NAME'].str.translate(str.maketrans('', '', string.punctuation))

# prep DNB DOMESTIC Ultimate columns, cast to upper case, remove punctuation
df['DNB_DOMESTIC_ULT_NUMBER'] = df['DNB_DOMESTIC_ULT_NUMBER'].astype('Int64', errors='ignore')
df['DNB_DOMESTIC_ULTIMATE_NAME'] = df['DNB_DOMESTIC_ULTIMATE_NAME'].str.upper()
df["DNB_DOMESTIC_ULTIMATE_NAME"] = df['DNB_DOMESTIC_ULTIMATE_NAME'].str.translate(str.maketrans('', '', string.punctuation))

# prep DNB HQ columns, case to upper case, remove punctuation
df['DNB_HQ_NUMBER'] = df['DNB_HQ_NUMBER'].astype('Int64', errors='ignore')
df['DNB_HQ_NAME'] = df['DNB_HQ_NAME'].str.upper()
df["DNB_HQ_NAME"] = df['DNB_HQ_NAME'].str.translate(str.maketrans('', '', string.punctuation))

df['DNB_CUSTOMER_NAME'] = np.nan
df['DNB_LEVEL'] = 'None'

# SET DNB_CUSTOMER_NAME and DNB_LEVEL to the highest non-null level available
df.loc[~df["DNB_GLOBAL_ULT_NAME"].isnull(),'DNB_LEVEL'] = "DUNS Global"
df.loc[~df["DNB_GLOBAL_ULT_NAME"].isnull(),'DNB_CUSTOMER_NAME'] = df.DNB_GLOBAL_ULT_NAME

df.loc[(df["DNB_GLOBAL_ULT_NAME"].isnull())&(~df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull()),'DNB_LEVEL'] = "DUNS Domestic"
df.loc[(df["DNB_GLOBAL_ULT_NAME"].isnull())&(~df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_DOMESTIC_ULTIMATE_NAME

df.loc[(df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull())&(~df["DNB_HQ_NAME"].isnull()),'DNB_LEVEL'] = "DUNS HQ"
df.loc[(df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull())&(~df["DNB_HQ_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_HQ_NAME

df.loc[(df["DNB_HQ_NAME"].isnull())&(~df["DNB_BUSINESS_NAME"].isnull()),'DNB_LEVEL'] = "DUNS"
df.loc[(df["DNB_HQ_NAME"].isnull())&(~df["DNB_BUSINESS_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_BUSINESS_NAME

#=======================
# Source: EDW

# default to None and Unknown
df['EDW_STATE'] = 'Unknown'
df.loc[df["EDW_CUSTOMER_NAME"].isnull(),'EDW_STATE'] = "None"
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'EDW_STATE'] = "Set"

# Retain the original untreated EDW Customer Name as well as the original account name
df['EDW_CUSTOMER_NAME_ORIGINAL'] = df['EDW_CUSTOMER_NAME']
df['CUSTOMER_ACCOUNT_NAME_ORIGINAL'] = df['CUSTOMER_ACCOUNT_NAME']

# remove tokens used to differentiate between customer names across conversions
df['EDW_CUSTOMER_NAME'].str.strip()
ending_tokens = [' 2', ' 3', ' 4', ' 04', ' 5', ' 6', ' 7', ' 8', ' 9',' (2)',
                 ' (3)',' (04)',' (4)', ' (5)', ' (6)', ' (7)', ' (8)',
                 ' (9)',' (25)','  (32)', ' AD', ' LD']

for s in ending_tokens:
    index_offset = -1*(len(s))
    df.loc[df['EDW_CUSTOMER_NAME'].str.endswith(s, na=False),"EDW_CUSTOMER_NAME"] = df['EDW_CUSTOMER_NAME'].str[:index_offset]

# do the same for account names
df['CUSTOMER_ACCOUNT_NAME'].str.strip()
for s in ending_tokens:
    index_offset = -1*(len(s))
    df.loc[df['CUSTOMER_ACCOUNT_NAME'].str.endswith(s, na=False),"CUSTOMER_ACCOUNT_NAME"] = df['CUSTOMER_ACCOUNT_NAME'].str[:index_offset]

#=======================

# SET PRIORITIES OF SOURCES
# Priority 1: EDW Customer
# Priority 2: DNB Customer
# Priority 3: MDM Customer
# Priority 4: ACCOUNT Name

df['CUSTOMER'] = np.nan
df['CUST_CALC_SOURCE'] = 'Unknown'
df['CUST_CALC_RULE'] = 'None'
# If we have an EDW Customer Name use it and set the Cust Calc Source to EDW
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'CUSTOMER'] = df["EDW_CUSTOMER_NAME"]
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'CUST_CALC_SOURCE'] = "EDW"

# Otherwise, if we have an DnB Customer Name use it and set the Cust Calc Source to DNB
df.loc[(df["CUSTOMER"].isnull())&(~df["DNB_CUSTOMER_NAME"].isnull()),'CUST_CALC_SOURCE'] = "DNB"
df.loc[df["CUSTOMER"].isnull(),'CUSTOMER'] = df["DNB_CUSTOMER_NAME"]

# Move the MDM this section to the following section in order to combine MDM matches into the new customer structure
#df.loc[df['CUSTOMER'].isnull(), 'CUST_CALC_SOURCE'] = 'MDM'
#df.loc[df['CUSTOMER'].isnull(), 'CUSTOMER'] = df['MDM_WEX_NAME']

# Otherwise, use the Account Name
df.loc[df["CUSTOMER"].isnull(),'CUST_CALC_SOURCE'] = 'ACCOUNT'
df.loc[df["CUSTOMER"].isnull(),'CUSTOMER'] = df["CUSTOMER_ACCOUNT_NAME"]

# reset the original EDW and Account Names
# this is done for QA purposes
# retaining the ability to see what the EDW Customer name as well as the account name look like in EDW
df['EDW_CUSTOMER_NAME'] = df['EDW_CUSTOMER_NAME_ORIGINAL']
df['CUSTOMER_ACCOUNT_NAME'] = df['CUSTOMER_ACCOUNT_NAME_ORIGINAL']

del(df['EDW_CUSTOMER_NAME_ORIGINAL'])
del(df['CUSTOMER_ACCOUNT_NAME_ORIGINAL'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df), "account rows")
print(len(df.CUSTOMER.unique()), "customer rows")
df.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#=======================
# Apply Custom Rules
# These Rules are modelled after the rules present in the legacy customer relationship report
# these rules only affect about 4,000 accounts nut they act very large customers
# in terms of cards/gallons/spend/revenue
# and in turn have been manually verified

# RULE SETs
def apply_rule_with_list(df, filter_name_list,final_name):

    # rule: all customer names with customer names in filter_name_list are replaced with final_name

    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUST_CALC_SOURCE"] = "CUSTOM RULE"
    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUSTOMER"] = final_name

    return(df)

def apply_rule_starts_with(df, compares_to, starts_with_string,final_name):

    # rule: all customer names with rows that have compares_to field value that starts with starts_with_string are replaced with final_name
    # compares_to field uses below include 'CUSTOMER' or 'DNB_CUSTOMER_NAME'

    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUST_CALC_SOURCE"] = "CUSTOM RULE"
    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUSTOMER"] = final_name

    return(df)

def apply_rule_contains(df, compares_to, contains_string,final_name):

    # rule: all rows with compares_to field value that contains contains_string are replaced with
    # this rule is not currently used below, these instances have been replaced with 'starts with rules'

    df.loc[df[compares_to].str.contains(contains_string, na=False),"CUST_CALC_SOURCE"] = "CUSTOM RULE"
    df.loc[df[compares_to].str.contains(contains_string, na=False),"CUSTOMER"] = final_name

    return(df)

df = apply_rule_with_list(df, ['QUANTA SERVICES INC','QUANTA SERVICES'], 'QUANTA SERVICES INC')
df = apply_rule_with_list(df, ['7325 ADVANCE STORES COMPANY 4','7325 ADVANCE AUTO','7325 ADVANCE STORES COMP','7325 ADVANCE STORES COMP 2'], 'ADVANCE AUTO')
df = apply_rule_with_list(df, ['17435-GE HEALTHCARE (3)','17435 GE HEALTHCARE','17435-GE HEALTHCARE','17435-GE HEALTHCARE (2)'], '17435-GE HEALTHCARE')
df = apply_rule_with_list(df, ['ADAPTHEALTH CORP','ADAPTHEALTH LLC'], 'ADAPTHEALTH')
df = apply_rule_with_list(df, ['AGL RESOURCES 5CU6','AGL RESOURCES ARI'], 'AGL RESOURCES')
df = apply_rule_with_list(df, ['APTIVE ENVIRONMENTAL LLC','ADAPTIVE ENVIRONMENTAL CONSULTING INC','APTIVE ENVIRONMENTAL LLC','APTIVE ENVIRONMENTAL 0CV7'], 'APTIVE ENVIRONMENTAL')
df = apply_rule_with_list(df, [' MMCCARTHY TIRE','MCCARTHY TIRE'], 'MCCARTHY TIRE')
df = apply_rule_with_list(df, ['FOSS NATIONAL/CORP-RATEFOSS NATIONAL LEASING (2)','FOSS NATIONAL LEASING (2)'], 'FOSS NATIONAL LEASING')
df = apply_rule_with_list(df, ['JOHNSONJOHNSON','JOHNSON JOHNSON CITRUS'], 'JOHNSON JOHNSON')
df = apply_rule_with_list(df, ['HELMERICH AND PAYNE INC PARENT','HELMERICH  PAYNE ID 5FM8'], 'HELMERICH AND PAYNE INC')
df = apply_rule_with_list(df, ['LABCORP','LABCORP (3LAB)','LABCORP (3LAB)(2)','LABORATORY CORPORATION OF AMERICA'], 'LABORATORY CORPORATION OF AMERICA')
df = apply_rule_with_list(df, ['LEHIGH HANSON (3LHN)','LEHIGH HANSON (3LHN)(2)','3995 LEHIGH HANSON INC','116710-LEHIGH HANSON, INC (2)',
                                 '116710-LEHIGH HANSON, INC (3)','116710-LEHIGH HANSON, INC (4)','116710-LEHIGH HANSON, INC (6)',
                                 "E450 LEHIGH HANSON MATERIALS L",'LEHIGH HANSON INC3LHC'], 'LEHIGH HANSON INC')
df = apply_rule_with_list(df, ['NOVONORDISK','NOVO NORDISK','NOVO NORDISK INC','NOVO NORDISK FONDEN'], 'NOVO NORDISK INC')
df = apply_rule_with_list(df, ['US LBM (5EE8)','US LBM (5EE8)(2)','US LBM HOLDINGS, LLC (5EE8)','LAMPERT YARDS US LBM LLC','US LBM HOLDINGS LLC 5EE8'], 'US LBM HOLDINGS LLC')
df = apply_rule_with_list(df, ['VEOLIA WATER LOGISTICS (2R63)','VEOLIA LOGISTICS 2R63'], 'VEOLIA LOGISTICS')

df = apply_rule_starts_with(df, 'CUSTOMER',"AT  T" , "AT&T")
df = apply_rule_starts_with(df, 'CUSTOMER',"FEDEX" , "FEDEX")
df = apply_rule_starts_with(df, 'CUSTOMER','ARAMARK', 'ARAMARK')
df = apply_rule_starts_with(df, 'CUSTOMER','CABLE ONE', 'CABLE ONE INC')
df = apply_rule_starts_with(df, 'CUSTOMER','CBRE', 'CBRE GROUP INC')
df = apply_rule_starts_with(df, 'CUSTOMER','COMCAST CABLE', 'COMCAST CABLE')
df = apply_rule_starts_with(df, 'CUSTOMER','COMPASS GROUP', 'COMPASS GROUP')
df = apply_rule_starts_with(df, 'CUSTOMER','CONOCOPHILLIPS', 'CONOCOPHILLIPS')
df = apply_rule_starts_with(df, 'CUSTOMER','CSC SERVICEWORKS', 'CSC SERVICEWORKS')
df = apply_rule_starts_with(df, 'CUSTOMER','CROWN CASTLE USA', 'CROWN CASTLE USA')
df = apply_rule_starts_with(df, 'CUSTOMER','E JOHNSON CONTROLS', 'E JOHNSON CONTROLS FIRE & SEC')
df = apply_rule_starts_with(df, 'CUSTOMER','FASTENAL', 'FASTENAL COMPANY')
df = apply_rule_starts_with(df, 'CUSTOMER','GENERAL MILLS', 'GENERAL MILLS')
df = apply_rule_starts_with(df, 'CUSTOMER','J R SIMPLOT', 'J R SIMPLOT')
df = apply_rule_starts_with(df, 'CUSTOMER','JC EHRLICH', 'JC EHRLICH')
df = apply_rule_starts_with(df, 'CUSTOMER','KINDER MORGAN', 'KINDER MORGAN')
df = apply_rule_starts_with(df, 'CUSTOMER','LIBERTY MUTUAL', 'LIBERTY MUTUAL')
df = apply_rule_starts_with(df, 'CUSTOMER','IGT GLOBAL', 'IGT GLOBAL')
df = apply_rule_starts_with(df, 'CUSTOMER','MARATHON PETROLEUM', 'MARATHON PETROLEUM')
df = apply_rule_starts_with(df, 'CUSTOMER','MONDELEZ GLOBAL', 'MONDELEZ GLOBAL')
df = apply_rule_starts_with(df, 'CUSTOMER','NATIONAL FUEL', 'NATIONAL FUEL')
df = apply_rule_starts_with(df, 'CUSTOMER','NEXSTAR BROADCASTING', 'NEXSTAR BROADCASTING')
df = apply_rule_starts_with(df, 'CUSTOMER','NORFOLK SOUTHERN', 'NORFOLK SOUTHERN')
df = apply_rule_starts_with(df, 'CUSTOMER','NORTHERN CLEARING', 'NORTHERN CLEARING')
df = apply_rule_starts_with(df, 'CUSTOMER','PHILLIPS 66 COMPANY', 'PHILLIPS 66 COMPANY')
df = apply_rule_starts_with(df, 'CUSTOMER','SCHINDLER ELEVATOR', 'SCHINDLER ELEVATOR')
df = apply_rule_starts_with(df, 'CUSTOMER','STONEMOR', 'STONEMOR')
df = apply_rule_starts_with(df, 'CUSTOMER','SYNGENTA', 'SYNGENTA')
df = apply_rule_starts_with(df, 'CUSTOMER','TRANSDEV', 'TRANSDEV')
df = apply_rule_starts_with(df, 'CUSTOMER','UNITED RENTALS', 'UNITED RENTALS INC')
df = apply_rule_starts_with(df, 'CUSTOMER','VAN POOL TRANSPORTATION', 'VAN POOL TRANSPORTATION')
df = apply_rule_starts_with(df, 'CUSTOMER','WILLIAMS STRATEGIC', 'WILLIAMS STRATEGIC')
df = apply_rule_starts_with(df, 'CUSTOMER','XTO ENERGY', 'XTO ENERGY')
df = apply_rule_starts_with(df, 'CUSTOMER',"BIMBO" , "BIMBO BAKERIES USA INC")

df = apply_rule_starts_with(df, 'CUSTOMER',"MANSFIELD OIL" , "MANSFIELD OIL")
df = apply_rule_starts_with(df, 'CUSTOMER',"ASPLUNDH" , "ASPLUNDH")
df = apply_rule_starts_with(df, 'CUSTOMER',"CINTAS CORPORATION" , "CINTAS CORPORATION")
df = apply_rule_starts_with(df, 'CUSTOMER',"ENTERPRISE RAC" , "ENTERPRISE RAC")
df = apply_rule_starts_with(df, 'CUSTOMER',"THE CRAWFORD GROUP INC" , "ENTERPRISE RAC")
df = apply_rule_starts_with(df, 'DNB_CUSTOMER_NAME',"STATE OF NEW YORK" , "STATE OF NEW YORK")
df = apply_rule_starts_with(df, 'DNB_CUSTOMER_NAME',"STATE OF GEORGIA" , "STATE OF GEORGIA")
df = apply_rule_starts_with(df, 'DNB_CUSTOMER_NAME',"VERIZON COMMUNICATIONS INC" , "VERIZON SOURCING LLC")
df = apply_rule_starts_with(df, 'DNB_CUSTOMER_NAME',"STATE OF NORTH CAROLINA" , "STATE OF NORTH CAROLINA")
df = apply_rule_starts_with(df, 'CUSTOMER',"DYCOM INDUSTRIES" , "DYCOM INDUSTRIES")
df = apply_rule_starts_with(df, 'CUSTOMER',"TERMINIX" , "TERMINIX CONSUMER SERVICES LLC")
df = apply_rule_starts_with(df, 'CUSTOMER',"WAYNE MUTUAL INSURANCE CO" , "WAYNE MUTUAL INSURANCE CO")

df = apply_rule_starts_with(df, 'CUSTOMER',"ART WATT PAINTING" , "ART WATT PAINTING")
df = apply_rule_starts_with(df, 'CUSTOMER',"BUENA VISTA SECURITY AND PROTE" , "BUENA VISTA SECURITY AND PROTE")
df = apply_rule_starts_with(df, 'CUSTOMER',"GREAT STONE GRANITE" , "GREAT STONE GRANITE")
df = apply_rule_starts_with(df, 'CUSTOMER',"SCHULTHEIS BROS CO" , "SCHULTHEIS BROS CO")
df = apply_rule_starts_with(df, 'CUSTOMER',"PIECE OF GREEN" , "PIECE OF GREEN LANDSCAPING")
df = apply_rule_starts_with(df, 'CUSTOMER',"CLEARPOINT CONSULTING ENGINEER" , "CLEARPOINT CONSULTING ENGINEER")
df = apply_rule_starts_with(df, 'CUSTOMER',"GRADY CRAWFORD CONSTRUCTION CO" , "GRADY CRAWFORD CONSTRUCTION CO")
df = apply_rule_starts_with(df, 'CUSTOMER',"ALLSOUTH APPLIANCE GROUP" , "ALLSOUTH APPLIANCE GROUP")
df = apply_rule_starts_with(df, 'CUSTOMER',"DAC ENTERPRISES" , "DAC ENTERPRISES")
df = apply_rule_starts_with(df, 'CUSTOMER',"HIGHLAND COMMUNITY COLLEGE" , "HIGHLAND COMMUNITY COLLEGE")
df = apply_rule_starts_with(df, 'CUSTOMER',"MATRIX SERVICE COMPANY" , "MATRIX SERVICE COMPANY")
df = apply_rule_starts_with(df, 'CUSTOMER',"TWIN CITY LIMOUSINES" , "TWIN CITY LIMOUSINES")
df = apply_rule_starts_with(df, 'CUSTOMER',"RMI SERVICES CORPORATION" , "RMI SERVICES CORPORATION")
df = apply_rule_starts_with(df, 'CUSTOMER',"GREAT LAKES HOME HEALTH S" , "GREAT LAKES HOME HEALTH S")
df = apply_rule_starts_with(df, 'CUSTOMER',"TOTAL DEPTH" , "TOTAL DEPTH")
df = apply_rule_starts_with(df, 'CUSTOMER',"CAPE BUILDING SYSTEMS" , "CAPE BUILDING SYSTEMS")
df = apply_rule_starts_with(df, 'CUSTOMER',"STATE OF TEXAS" , "STATE OF TEXAS")
df = apply_rule_starts_with(df, 'CUSTOMER',"RICS ELECTRIC" , "RICS ELECTRIC")
df = apply_rule_starts_with(df, 'CUSTOMER',"GILS CARPETS" , "GILS CARPETS")
df = apply_rule_starts_with(df, 'CUSTOMER',"PREMIER THERAPY" , "PREMIER THERAPY")
df = apply_rule_starts_with(df, 'CUSTOMER',"STE GENEVIEVE COUNTY MEMORIAL" , "STE GENEVIEVE COUNTY MEMORIAL")
df = apply_rule_starts_with(df, 'CUSTOMER',"STILSING ELECTRIC INC" , "STILSING ELECTRIC INC")
df = apply_rule_starts_with(df, 'CUSTOMER',"ROBINSON AUTO SALES" , "ROBINSON AUTO SALES")
df = apply_rule_starts_with(df, 'CUSTOMER',"FLORIDA FIRST CALL REMOVAL SER" , "FLORIDA FIRST CALL REMOVAL SER")
df = apply_rule_starts_with(df, 'CUSTOMER',"LANDMARK CONSTRUCTION SOLUTIO" , "LANDMARK CONSTRUCTION SOLUTIO")
df = apply_rule_starts_with(df, 'CUSTOMER',"IMPORT MOTORS OF OLD SAYBROOK" , "IMPORT MOTORS OF OLD SAYBROOK")
df = apply_rule_starts_with(df, 'CUSTOMER',"RALEIGH EAST CONCRETE CONSTRUC" , "RALEIGH EAST CONCRETE CONSTRUC")
df = apply_rule_starts_with(df, 'CUSTOMER',"AMES PLUMBING SERVIC" , "AMES PLUMBING SERVIC")
df = apply_rule_starts_with(df, 'CUSTOMER',"FENIGOR GROUP LLC" , "FENIGOR GROUP LLC")
df = apply_rule_starts_with(df, 'CUSTOMER',"MONTCALM COUNTY ROAD COMMISSIO" , "MONTCALM COUNTY ROAD COMMISSIO")
df = apply_rule_starts_with(df, 'CUSTOMER',"WESTERN INDUSTRIAL CONTRACTORS" , "WESTERN INDUSTRIAL CONTRACTORS")
df = apply_rule_starts_with(df, 'CUSTOMER',"TIRE CORRAL OF AMERICA" , "TIRE CORRAL OF AMERICA")
df = apply_rule_starts_with(df, 'CUSTOMER',"CARVANA" , "CARVANA")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df), "account rows")
print(len(df.CUSTOMER.unique()), "customer rows")
df.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# RDW Conversions tracked by the conversion team
# remove leading and trailing blanks, remove dashes
# convert to Int64
# df_r columns: FLEET_ID, CLASSIC_ACCOUNT_NUMBER

df_r = RDW_CONVERSIONS_df[['FLEET_ID','CLASSIC_ACCOUNT_NUMBER']].dropna(subset=['CLASSIC_ACCOUNT_NUMBER']).copy()
df_r.FLEET_ID = df_r.FLEET_ID.str.strip()
df_r = df_r[~df_r['FLEET_ID'].str.contains('-',na=False)]
df_r.FLEET_ID = df_r.FLEET_ID.astype('float')
df_r.FLEET_ID = df_r.FLEET_ID.astype('Int64')
print(len(df_r), "total RDW conversions")

# create a copy of a subset of columns of customer hierarchy calculated up to this point
# columns: CUSTOMER_ACCOUNT_ID, CUSTOMER
df_cust = df[['CUSTOMER_ACCOUNT_ID','CUSTOMER']].copy()
df_cust.CUSTOMER_ACCOUNT_ID = df_cust['CUSTOMER_ACCOUNT_ID'].astype('Int64')

# join df_cust onto df_r
# in order to add CUSTOMER Name to the RDW Conversion set
# columns: FLEET_ID, CLASSIC_ACCOUNT_NUMBER, CUSTOMER_ACCOUNT_ID, CUSTOMER
df_rj = pd.merge(df_r, df_cust, left_on='FLEET_ID', right_on='CUSTOMER_ACCOUNT_ID', how='inner')
df_rj = df_rj[pd.to_numeric(df_rj.CLASSIC_ACCOUNT_NUMBER, errors='coerce').notnull()]
df_rj.CLASSIC_ACCOUNT_NUMBER = df_rj.CLASSIC_ACCOUNT_NUMBER.astype(float)
df_rj.CLASSIC_ACCOUNT_NUMBER = df_rj.CLASSIC_ACCOUNT_NUMBER.astype(np.int64)

# create a second copy of the subset of columns of customer hierarchy calculated up to this point
df_cust_classic = df[['CUSTOMER_ACCOUNT_ID','CUSTOMER']].copy()
df_cust_classic.columns = ['CUSTOMER_ACCOUNT_ID', 'CLASSIC_CUSTOMER']
df_cust_classic.CUSTOMER_ACCOUNT_ID = df_cust['CUSTOMER_ACCOUNT_ID'].astype('Int64')

# join this structure to the RDW conversion dataset
# so that we now have both the new acount id, the old account id
# as well as the new CUSTOMER Name and the old CUSTOMER NAME
df_rj = df_rj[['FLEET_ID','CLASSIC_ACCOUNT_NUMBER','CUSTOMER']]
df_rj = pd.merge(df_rj, df_cust_classic, left_on='CLASSIC_ACCOUNT_NUMBER', right_on='CUSTOMER_ACCOUNT_ID', how='inner')

# we are interested in those conversion cases
# where we see a different before customer name compared to the after customer name
df_rdw_conversions = df_rj[df_rj.CUSTOMER!=df_rj.CLASSIC_CUSTOMER]
print(len(df_rdw_conversions), "unhandled conversions from RDW")

# what we have left is a dataframe containing those accounts
# that need to have their old classic customer name replaced with the new customer name
# columns: CUSTOMER_ACCOUNT_ID, CONVERSION_REPLACEMENT_CUSTOMER

df_rdw_conversions = df_rdw_conversions[['CUSTOMER_ACCOUNT_ID','CUSTOMER']]
df_rdw_conversions.columns = ['CUSTOMER_ACCOUNT_ID','CONVERSION_REPLACEMENT_CUSTOMER']
df_rdw_conversions.drop_duplicates(subset=['CUSTOMER_ACCOUNT_ID'], inplace=True)

print(len(df), "before join")
df_j = pd.merge(df, df_rdw_conversions, on='CUSTOMER_ACCOUNT_ID',how='left')
print(len(df_j), "after join")

# replace the CUSTOMER field with the contents of the CONVERSION_REPLACEMENT_CUSTOMER where this is not null
# and track the calculation source
df_j.loc[~df_j["CONVERSION_REPLACEMENT_CUSTOMER"].isnull(),'CUSTOMER'] = df_j.CONVERSION_REPLACEMENT_CUSTOMER
df_j.loc[~df_j["CONVERSION_REPLACEMENT_CUSTOMER"].isnull(),'CUST_CALC_SOURCE'] = 'RDW CONVERSIONS'

# remove the temporary conversion replacement name column
del(df_j['CONVERSION_REPLACEMENT_CUSTOMER'])
df_j.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df_j), "account rows")
print(len(df_j.CUSTOMER.unique()), "customer rows")
df_j.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# incorporate un matched rows from MDM
# New MDM matches, shared by Wes Corbin during the week of Nov 17, 2022

#print(len(ACCOUNTS_PARTY_EXTRACT_df), 'MDM account rows')
#df_mdm = ACCOUNTS_PARTY_EXTRACT_df[['ACCOUNTNUMBER','WEXBUSINESSID','NAME']].copy()
#df_mdm.columns = ['CUSTOMER_ACCOUNT_ID','WEX_BUSINESS_ID','WEX_BUSINESS_NAME']
#print(len(df_mdm))
#df_mdm.dropna(subset=['CUSTOMER_ACCOUNT_ID'], inplace=True)
#print(len(df_mdm))

#df_mdm['WEX_BUSINESS_NAME'] = df_mdm['WEX_BUSINESS_NAME'].str.upper()
#df_mdm['WEX_BUSINESS_NAME'] = df_mdm['WEX_BUSINESS_NAME'].str.translate(str.maketrans('','', string.punctuation))

# filter out known non-customer entities expressed in MDM
#df_mdm = df_mdm[df_mdm.WEX_BUSINESS_NAME!='CARD TYPE 7 PRIMARY']
#df_mdm = df_mdm[df_mdm.WEX_BUSINESS_NAME!='ELEMENT 1']
#df_mdm = df_mdm[df_mdm.WEX_BUSINESS_NAME!='ELEMENT 2']
#print(len(df_mdm), "MDM account rows after filter rules")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df_j_with_mdm = pd.merge(df_j, df_mdm, on='CUSTOMER_ACCOUNT_ID', how='left')
#df_j_with_mdm.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df_j = pd.merge(df, ACCOUNTS_WITH_EBX_PARTY_df, on='CUSTOMER_ACCOUNT_ID', how='left')
#df_j.loc[~df_j["PARTY_DEFAULT_NAME"].isnull(),'CUSTOMER'] = df_j.PARTY_DEFAULT_NAME
#df_j.loc[~df_j["PARTY_DEFAULT_NAME"].isnull(),'CUST_CALC_SOURCE'] = 'MDM'
#df_j.CUST_CALC_SOURCE.value_counts()

#del(df_j['PARTY_ID'])
#del(df_j['PARTY_DEFAULT_NAME'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#print(len(df_matches_verified))
#df_matches_verified.drop_duplicates(subset='CUSTOMER', inplace=True)
#print(len(df_matches_verified))

#print(len(df_j))
#df_j_w_verified = pd.merge(df_j, df_matches_verified, left_on='CUSTOMER', right_on='CUSTOMER', how='left')
#print(len(df_j_w_verified))

#df_j_w_verified.loc[~df_j_w_verified["CUSTOMER_CLC"].isnull(),'CUSTOMER'] = df_j_w_verified.CUSTOMER_CLC
#df_j_w_verified.loc[~df_j_w_verified["CUSTOMER_CLC"].isnull(),'CUST_CALC_SOURCE'] = 'CLC'

#df_j = df_j_w_verified
#del(df_j['CUSTOMER_CLC'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df_j.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#RDW_CONVERSIONS_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df_j.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#unique_customer_list = df_j.CUSTOMER.unique()
#df_customer_ids = pd.DataFrame(unique_customer_list)
#df_customer_ids.columns = ["CUSTOMER"]
#df_customer_ids = df_customer_ids.sort_values(['CUSTOMER']).reset_index(drop=True)
#df_customer_ids = df_customer_ids.reset_index(drop=False)
#df_customer_ids['CUSTOMER_ID'] = df_customer_ids.index + 77000000
#del(df_customer_ids['index'])
#df_customer_ids.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#print(len(df_j))
#df_jj = pd.merge(df_j, df_customer_ids, on='CUSTOMER')
#df_jj.dropna(subset=['CUSTOMER'], inplace=True)
#print(len(df_jj))
#df_jj.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df_by_account = df_jj[['CUSTOMER_ACCOUNT_ID','CUSTOMER_ID', 'CUSTOMER']]
#df_by_account.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#len(df_by_account.CUSTOMER_ID.unique())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df_jj.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#ACCOUNT_NEW_SALES_FULL_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#ACCOUNT_NEW_SALES_FULL_df.columns.tolist()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#print(len(df_jj))
#print(len(ACCOUNT_NEW_SALES_df))
#ACCOUNT_NEW_SALES_df['HAS_SALES_FLAG'] = True
#df_j_with_sales = pd.merge(df_jj, ACCOUNT_NEW_SALES_df, on='CUSTOMER_ACCOUNT_ID', how='left')
#print(len(df_j_with_sales))

#print(len(df_jj))
#print(len(ACCOUNT_NEW_SALES_FULL_df))
#ACCOUNT_NEW_SALES_FULL_df.columns = ['SALES_MARKETING_PARTNER_NM','SALES_BUSINESS_PROGRAM_NM','SALES_PROGRAM_ID','CUSTOMER_ACCOUNT_ID','SALES_CAMPAIGN_TYPE','SALES_COUPON_CODE','SALES_CHANNEL','SALES_REP','SALES_TRANS_RECORDS','SALES_DATA_SOURCE','HAS_SALES_FLAG']
#ACCOUNT_NEW_SALES_FULL_df['HAS_SALES_FLAG'] = True
#df_j_with_sales = pd.merge(df_jj, ACCOUNT_NEW_SALES_FULL_df, on='CUSTOMER_ACCOUNT_ID', how='left')
#print(len(df_j_with_sales))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df_j_with_sales.HAS_SALES_FLAG.value_counts(dropna=False)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df = df_j_with_sales

# Write recipe outputs
#ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS")
#ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS.write_with_schema(ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df)

#BY_ACCOUNT = dataiku.Dataset("BY_ACCOUNT")
#BY_ACCOUNT.write_with_schema(df_by_account)
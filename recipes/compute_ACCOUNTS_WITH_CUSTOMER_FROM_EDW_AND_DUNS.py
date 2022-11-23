# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import string

# Read recipe inputs
ACCOUNTS_WITH_BUNDLER_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_BUNDLER_AND_DUNS")
ACCOUNTS_WITH_BUNDLER_AND_DUNS_df = ACCOUNTS_WITH_BUNDLER_AND_DUNS.get_dataframe()

ACCOUNTS_WITH_EBX_PARTY = dataiku.Dataset("ACCOUNTS_WITH_EBX_PARTY")
ACCOUNTS_WITH_EBX_PARTY_df = ACCOUNTS_WITH_EBX_PARTY.get_dataframe()

ACCOUNTS_PARTY_EXTRACT = dataiku.Dataset("Account_Party_extract")
ACCOUNTS_PARTY_EXTRACT_df = ACCOUNTS_PARTY_EXTRACT.get_dataframe()

MDM_FINAL = dataiku.Dataset("mdm_final")
MDM_FINAL_df = MDM_FINAL.get_dataframe()

MATCHES_VERIFIED = dataiku.Dataset("VERIFIED_MATCHES")
MATCHES_VERIFIED_df = MATCHES_VERIFIED.get_dataframe()

RDW_CONVERSIONS = dataiku.Dataset("NAFCUSTOMER_RDW_CONVERSIONS")
RDW_CONVERSIONS_df = RDW_CONVERSIONS.get_dataframe()

ACCOUNT_NEW_SALES = dataiku.Dataset("ACCOUNT_NEW_SALES")
ACCOUNT_NEW_SALES_df = ACCOUNT_NEW_SALES.get_dataframe()

ACCOUNT_NEW_SALES_FULL = dataiku.Dataset("ACCOUNT_NEW_SALES_FULL")
ACCOUNT_NEW_SALES_FULL_df = ACCOUNT_NEW_SALES_FULL.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# load matches calculated by matching process, then verified with secondary matching step

#df_matches_verified = MATCHES_VERIFIED_df
#df_matches_verified["CUSTOMER"] = df_matches_verified['CUSTOMER'].str.translate(str.maketrans('', '', string.punctuation))
#df_matches_verified["CUSTOMER_CLC"] = df_matches_verified['CUSTOMER_CLC'].str.translate(str.maketrans('', '', string.punctuation))
#df_matches_verified.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(RDW_CONVERSIONS_df))
RDW_CONVERSIONS_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# load MDM matches, note we are using an older MDM Final extract from March
# TODO: replace MDM final with snowflake source

# prepare mdm final output:
# retain key columns
# cast ids to large ints
# remove punctuation cast names upper case
# filter out invalid grouping entries

df_mdm = MDM_FINAL_df[['accountnumber','wex_id','name','global_customer_id','global_customer_name']].copy()
df_mdm.columns = ['CUSTOMER_ACCOUNT_ID','MDM_WEX_ID','MDM_WEX_NAME','MDM_PARTY_ID','MDM_PARTY_NAME']
df_mdm.dropna(subset=['CUSTOMER_ACCOUNT_ID'], inplace=True)
df_mdm['CUSTOMER_ACCOUNT_ID'] = df_mdm['CUSTOMER_ACCOUNT_ID'].astype('Int64', errors='ignore')
df_mdm['MDM_WEX_ID'] = df_mdm['MDM_WEX_ID'].astype('Int64', errors='ignore')
df_mdm['MDM_PARTY_ID'] = df_mdm['MDM_PARTY_ID'].astype('Int64', errors='ignore')
print(len(df_mdm))

df_mdm['MDM_WEX_NAME'] = df_mdm['MDM_WEX_NAME'].str.upper()
df_mdm["MDM_WEX_NAME"] = df_mdm['MDM_WEX_NAME'].str.translate(str.maketrans('', '', string.punctuation))

print(len(df_mdm))
df_mdm = df_mdm[df_mdm.MDM_WEX_NAME!='CARD TYPE 7 PRIMARY']
df_mdm = df_mdm[df_mdm.MDM_WEX_NAME!='ELEMENT 1']
df_mdm = df_mdm[df_mdm.MDM_WEX_NAME!='ELEMENT 2']
print(len(df_mdm))

df_mdm['MDM_PARTY_NAME'] = df_mdm['MDM_PARTY_NAME'].str.upper()
df_mdm["MDM_PART_NAME"] = df_mdm['MDM_PARTY_NAME'].str.translate(str.maketrans('', '', string.punctuation))
df_mdm.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# create structure at account level with MDM WEX ID and MDM WEX NAME
# that can be joined to source accounts

df_wex_id = df_mdm.groupby(['MDM_WEX_ID', 'MDM_WEX_NAME'])[['CUSTOMER_ACCOUNT_ID']].count().reset_index()
df_wex_id.columns = ['MDM_WEX_ID','MDM_WEX_NAME','COUNT_OF_ACCOUNT']
df_wex_id = df_wex_id[df_wex_id.COUNT_OF_ACCOUNT>1]
df_wex_id['HAS_WEX_ID'] = True
df_wex_id.sort_values(by=['COUNT_OF_ACCOUNT'],ascending=False, inplace=True)

df_wex_id.tail()
print(len(df_wex_id))
df_wex_id.head()

df_account_with_wex_id = df_mdm[['CUSTOMER_ACCOUNT_ID','MDM_WEX_ID']].copy()
df_account_with_wex_id.dropna(subset=['MDM_WEX_ID'], inplace=True)
print(len(df_account_with_wex_id))
df_account_with_wex_id = pd.merge(df_account_with_wex_id, df_wex_id, on='MDM_WEX_ID', how='left')
print(len(df_account_with_wex_id))
df_account_with_wex_id.head()
df_account_with_wex_id.dropna(subset=['HAS_WEX_ID'], inplace=True)
df_account_with_wex_id.drop_duplicates(subset=['CUSTOMER_ACCOUNT_ID'], inplace=True)
df_account_with_wex_id = df_account_with_wex_id[['CUSTOMER_ACCOUNT_ID','MDM_WEX_ID','MDM_WEX_NAME']]

print(len(df_account_with_wex_id))
df_account_with_wex_id.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Join accounts with MDM grouping to accounts
# ensure that counts before and after match, there should be no row duplication
df = ACCOUNTS_WITH_BUNDLER_AND_DUNS_df.copy()
print(len(df))

df_account_with_wex_id['CUSTOMER_ACCOUNT_ID'] = df_account_with_wex_id['CUSTOMER_ACCOUNT_ID'].astype('Int64', errors='ignore')
df = pd.merge(df,df_account_with_wex_id, on='CUSTOMER_ACCOUNT_ID', how='left')
print(len(df))
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import warnings
warnings.filterwarnings(action='once')

df['DUNS'] = df['DUNS'].astype('Int64', errors='ignore')
df['DNB_DUNS_NUMBER'] = df['DNB_DUNS_NUMBER'].astype('Int64', errors='ignore')
df['DNB_BUSINESS_NAME'] = df['DNB_BUSINESS_NAME'].str.upper()
df["DNB_BUSINESS_NAME"] = df['DNB_BUSINESS_NAME'].str.translate(str.maketrans('', '', string.punctuation))

df['DNB_GLOBAL_ULT_NUMBER'] = df['DNB_GLOBAL_ULT_NUMBER'].astype('Int64', errors='ignore')
df['DNB_GLOBAL_ULT_NAME'] = df['DNB_GLOBAL_ULT_NAME'].str.upper()
df["DNB_GLOBAL_ULT_NAME"] = df['DNB_GLOBAL_ULT_NAME'].str.translate(str.maketrans('', '', string.punctuation))

df['DNB_DOMESTIC_ULT_NUMBER'] = df['DNB_DOMESTIC_ULT_NUMBER'].astype('Int64', errors='ignore')
df['DNB_DOMESTIC_ULTIMATE_NAME'] = df['DNB_DOMESTIC_ULTIMATE_NAME'].str.upper()
df["DNB_DOMESTIC_ULTIMATE_NAME"] = df['DNB_DOMESTIC_ULTIMATE_NAME'].str.translate(str.maketrans('', '', string.punctuation))

df['DNB_HQ_NUMBER'] = df['DNB_HQ_NUMBER'].astype('Int64', errors='ignore')
df['DNB_HQ_NAME'] = df['DNB_HQ_NAME'].str.upper()
df["DNB_HQ_NAME"] = df['DNB_HQ_NAME'].str.translate(str.maketrans('', '', string.punctuation))

df['DNB_LEVEL'] = 'None'
df['DNB_CUSTOMER_NAME'] = np.nan

df.loc[~df["DNB_GLOBAL_ULT_NAME"].isnull(),'DNB_LEVEL'] = "DUNS Global"
df.loc[~df["DNB_GLOBAL_ULT_NAME"].isnull(),'DNB_CUSTOMER_NAME'] = df.DNB_GLOBAL_ULT_NAME

df.loc[(df["DNB_GLOBAL_ULT_NAME"].isnull())&(~df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull()),'DNB_LEVEL'] = "DUNS Domestic"
df.loc[(df["DNB_GLOBAL_ULT_NAME"].isnull())&(~df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_DOMESTIC_ULTIMATE_NAME

df.loc[(df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull())&(~df["DNB_HQ_NAME"].isnull()),'DNB_LEVEL'] = "DUNS HQ"
df.loc[(df["DNB_DOMESTIC_ULTIMATE_NAME"].isnull())&(~df["DNB_HQ_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_HQ_NAME

df.loc[(df["DNB_HQ_NAME"].isnull())&(~df["DNB_BUSINESS_NAME"].isnull()),'DNB_LEVEL'] = "DUNS"
df.loc[(df["DNB_HQ_NAME"].isnull())&(~df["DNB_BUSINESS_NAME"].isnull()),'DNB_CUSTOMER_NAME'] = df.DNB_BUSINESS_NAME

df['EDW_STATE'] = 'Unknown'
df.loc[df["EDW_CUSTOMER_NAME"].isnull(),'EDW_STATE'] = "None"
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'EDW_STATE'] = "Set"

df['EDW_CUSTOMER_NAME_ORIGINAL'] = df['EDW_CUSTOMER_NAME']
df['CUSTOMER_ACCOUNT_NAME_ORIGINAL'] = df['CUSTOMER_ACCOUNT_NAME']

df['EDW_CUSTOMER_NAME'].str.strip()
ending_tokens = [' 2', ' 3', ' 4', ' 04', ' 5', ' 6', ' 7', ' 8', ' 9',' (2)',
                 ' (3)',' (04)',' (4)', ' (5)', ' (6)', ' (7)', ' (8)',
                 ' (9)',' (25)','  (32)', ' AD', ' LD']

for s in ending_tokens:
    index_offset = -1*(len(s))
    df.loc[df['EDW_CUSTOMER_NAME'].str.endswith(s, na=False),"EDW_CUSTOMER_NAME"] = df['EDW_CUSTOMER_NAME'].str[:index_offset]

df['CUSTOMER_ACCOUNT_NAME'].str.strip()
for s in ending_tokens:
    index_offset = -1*(len(s))
    df.loc[df['CUSTOMER_ACCOUNT_NAME'].str.endswith(s, na=False),"CUSTOMER_ACCOUNT_NAME"] = df['CUSTOMER_ACCOUNT_NAME'].str[:index_offset]

df['CUSTOMER'] = np.nan
df['CUST_CALC_SOURCE'] = 'Unknown'
df['CUST_CALC_RULE'] = 'None'
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'CUSTOMER'] = df["EDW_CUSTOMER_NAME"]
df.loc[~df["EDW_CUSTOMER_NAME"].isnull(),'CUST_CALC_SOURCE'] = "EDW"

df.loc[(df["CUSTOMER"].isnull())&(~df["DNB_CUSTOMER_NAME"].isnull()),'CUST_CALC_SOURCE'] = "DNB"
df.loc[df["CUSTOMER"].isnull(),'CUSTOMER'] = df["DNB_CUSTOMER_NAME"]

df.loc[df['CUSTOMER'].isnull(), 'CUST_CALC_SOURCE'] = 'MDM'
df.loc[df['CUSTOMER'].isnull(), 'CUSTOMER'] = df['MDM_WEX_NAME']

df.loc[df["CUSTOMER"].isnull(),'CUST_CALC_SOURCE'] = 'ACCOUNT'
df.loc[df["CUSTOMER"].isnull(),'CUSTOMER'] = df["CUSTOMER_ACCOUNT_NAME"]

# RULE SETs
def apply_rule(df, rule_name,filter_name_list,final_name):

    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUST_CALC_SOURCE"] = "CUSTOM RULE"
    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUST_CALC_RULE"] = rule_name
    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUSTOMER"] = final_name

    return(df)

def apply_rule_starts_with(df, rule_name, compares_to, starts_with_string,final_name):

    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUST_CALC_SOURCE"] = "CUSTOM RULE"
    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUST_CALC_RULE"] = rule_name
    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUSTOMER"] = final_name

    return(df)

def apply_rule_contains(df, rule_name, compares_to, contains_string,final_name):

    df.loc[df[compares_to].str.contains(contains_string, na=False),"CUST_CALC_SOURCE"] = "CUSTOM RULE"
    df.loc[df[compares_to].str.contains(contains_string, na=False),"CUST_CALC_RULE"] = rule_name
    df.loc[df[compares_to].str.contains(contains_string, na=False),"CUSTOMER"] = final_name

    return(df)

# The following set of ~100 rules represents high card count mappings
# these rules have been manually verified
# as they impact large numbers of cards (and in turn gallons, spend and revenue)

df = apply_rule(df, "RULE 001", ['QUANTA SERVICES INC','QUANTA SERVICES'], 'QUANTA SERVICES INC')
df = apply_rule(df, "RULE 002", ['7325 ADVANCE STORES COMPANY 4','7325 ADVANCE AUTO','7325 ADVANCE STORES COMP','7325 ADVANCE STORES COMP 2'], 'ADVANCE AUTO')
df = apply_rule(df, "RULE 003", ['17435-GE HEALTHCARE (3)','17435 GE HEALTHCARE','17435-GE HEALTHCARE','17435-GE HEALTHCARE (2)'], '17435-GE HEALTHCARE')
df = apply_rule(df, "RULE 004", ['ADAPTHEALTH CORP','ADAPTHEALTH LLC'], 'ADAPTHEALTH')
df = apply_rule(df, "RULE 005", ['AGL RESOURCES 5CU6','AGL RESOURCES ARI'], 'AGL RESOURCES')
df = apply_rule(df, "RULE 006", ['APTIVE ENVIRONMENTAL LLC','ADAPTIVE ENVIRONMENTAL CONSULTING INC','APTIVE ENVIRONMENTAL LLC','APTIVE ENVIRONMENTAL 0CV7'], 'APTIVE ENVIRONMENTAL')
df = apply_rule(df, "RULE 007", [' MMCCARTHY TIRE','MCCARTHY TIRE'], 'MCCARTHY TIRE')
df = apply_rule(df, "RULE 008", ['FOSS NATIONAL/CORP-RATEFOSS NATIONAL LEASING (2)','FOSS NATIONAL LEASING (2)'], 'FOSS NATIONAL LEASING')
df = apply_rule(df, "RULE 009", ['JOHNSONJOHNSON','JOHNSON JOHNSON CITRUS'], 'JOHNSON JOHNSON')
df = apply_rule(df, "RULE 010", ['HELMERICH AND PAYNE INC PARENT','HELMERICH  PAYNE ID 5FM8'], 'HELMERICH AND PAYNE INC')
df = apply_rule(df, "RULE 011", ['LABCORP','LABCORP (3LAB)','LABCORP (3LAB)(2)','LABORATORY CORPORATION OF AMERICA'], 'LABORATORY CORPORATION OF AMERICA')
df = apply_rule(df, "RULE 012", ['LEHIGH HANSON (3LHN)','LEHIGH HANSON (3LHN)(2)','3995 LEHIGH HANSON INC','116710-LEHIGH HANSON, INC (2)',
                                 '116710-LEHIGH HANSON, INC (3)','116710-LEHIGH HANSON, INC (4)','116710-LEHIGH HANSON, INC (6)',
                                 "E450 LEHIGH HANSON MATERIALS L",'LEHIGH HANSON INC3LHC'], 'LEHIGH HANSON INC')
df = apply_rule(df, "RULE 013", ['NOVONORDISK','NOVO NORDISK','NOVO NORDISK INC','NOVO NORDISK FONDEN'], 'NOVO NORDISK INC')
df = apply_rule(df, "RULE 014", ['US LBM (5EE8)','US LBM (5EE8)(2)','US LBM HOLDINGS, LLC (5EE8)','LAMPERT YARDS US LBM LLC','US LBM HOLDINGS LLC 5EE8'], 'US LBM HOLDINGS LLC')
df = apply_rule(df, "RULE 015", ['VEOLIA WATER LOGISTICS (2R63)','VEOLIA LOGISTICS 2R63'], 'VEOLIA LOGISTICS')

df = apply_rule_starts_with(df, "RULE 016",'CUSTOMER',"AT  T" , "AT&T")
df = apply_rule_starts_with(df, "RULE 017",'CUSTOMER',"FEDEX" , "FEDEX")
df = apply_rule_starts_with(df, "RULE 018",'CUSTOMER','ARAMARK', 'ARAMARK')
df = apply_rule_starts_with(df, "RULE 019",'CUSTOMER','CABLE ONE', 'CABLE ONE INC')
df = apply_rule_starts_with(df, "RULE 020",'CUSTOMER','CBRE', 'CBRE GROUP INC')
df = apply_rule_starts_with(df, "RULE 021",'CUSTOMER','COMCAST CABLE', 'COMCAST CABLE')
df = apply_rule_starts_with(df, "RULE 022",'CUSTOMER','COMPASS GROUP', 'COMPASS GROUP')
df = apply_rule_starts_with(df, "RULE 023",'CUSTOMER','CONOCOPHILLIPS', 'CONOCOPHILLIPS')
df = apply_rule_starts_with(df, "RULE 024",'CUSTOMER','CSC SERVICEWORKS', 'CSC SERVICEWORKS')
df = apply_rule_starts_with(df, "RULE 025",'CUSTOMER','CROWN CASTLE USA', 'CROWN CASTLE USA')
df = apply_rule_starts_with(df, "RULE 026",'CUSTOMER','E JOHNSON CONTROLS', 'E JOHNSON CONTROLS FIRE & SEC')
df = apply_rule_starts_with(df, "RULE 027",'CUSTOMER','FASTENAL', 'FASTENAL COMPANY')
df = apply_rule_starts_with(df, "RULE 028",'CUSTOMER','GENERAL MILLS', 'GENERAL MILLS')
df = apply_rule_starts_with(df, "RULE 029",'CUSTOMER','J R SIMPLOT', 'J R SIMPLOT')
df = apply_rule_starts_with(df, "RULE 030",'CUSTOMER','JC EHRLICH', 'JC EHRLICH')
df = apply_rule_starts_with(df, "RULE 031",'CUSTOMER','KINDER MORGAN', 'KINDER MORGAN')
df = apply_rule_starts_with(df, "RULE 032",'CUSTOMER','LIBERTY MUTUAL', 'LIBERTY MUTUAL')
df = apply_rule_starts_with(df, "RULE 033",'CUSTOMER','IGT GLOBAL', 'IGT GLOBAL')
df = apply_rule_starts_with(df, "RULE 034",'CUSTOMER','MARATHON PETROLEUM', 'MARATHON PETROLEUM')
df = apply_rule_starts_with(df, "RULE 035",'CUSTOMER','MONDELEZ GLOBAL', 'MONDELEZ GLOBAL')
df = apply_rule_starts_with(df, "RULE 036",'CUSTOMER','NATIONAL FUEL', 'NATIONAL FUEL')
df = apply_rule_starts_with(df, "RULE 037",'CUSTOMER','NEXSTAR BROADCASTING', 'NEXSTAR BROADCASTING')
df = apply_rule_starts_with(df, "RULE 038",'CUSTOMER','NORFOLK SOUTHERN', 'NORFOLK SOUTHERN')
df = apply_rule_starts_with(df, "RULE 039",'CUSTOMER','NORTHERN CLEARING', 'NORTHERN CLEARING')
df = apply_rule_starts_with(df, "RULE 040",'CUSTOMER','PHILLIPS 66 COMPANY', 'PHILLIPS 66 COMPANY')
df = apply_rule_starts_with(df, "RULE 041",'CUSTOMER','SCHINDLER ELEVATOR', 'SCHINDLER ELEVATOR')
df = apply_rule_starts_with(df, "RULE 042",'CUSTOMER','STONEMOR', 'STONEMOR')
df = apply_rule_starts_with(df, "RULE 043",'CUSTOMER','SYNGENTA', 'SYNGENTA')
df = apply_rule_starts_with(df, "RULE 044",'CUSTOMER','TRANSDEV', 'TRANSDEV')
df = apply_rule_starts_with(df, "RULE 045",'CUSTOMER','UNITED RENTALS', 'UNITED RENTALS INC')
df = apply_rule_starts_with(df, "RULE 046",'CUSTOMER','VAN POOL TRANSPORTATION', 'VAN POOL TRANSPORTATION')
df = apply_rule_starts_with(df, "RULE 047",'CUSTOMER','WILLIAMS STRATEGIC', 'WILLIAMS STRATEGIC')
df = apply_rule_starts_with(df, "RULE 048",'CUSTOMER','XTO ENERGY', 'XTO ENERGY')
df = apply_rule_starts_with(df, "RULE 049",'CUSTOMER',"BIMBO" , "BIMBO BAKERIES USA INC")

df = apply_rule_starts_with(df, "RULE 050",'CUSTOMER',"MANSFIELD OIL" , "MANSFIELD OIL")
df = apply_rule_starts_with(df, "RULE 051",'CUSTOMER',"ASPLUNDH" , "ASPLUNDH")
df = apply_rule_starts_with(df, "RULE 052",'CUSTOMER',"CINTAS CORPORATION" , "CINTAS CORPORATION")
df = apply_rule_starts_with(df, "RULE 053",'CUSTOMER',"ENTERPRISE RAC" , "ENTERPRISE RAC")
df = apply_rule_starts_with(df, "RULE 054",'CUSTOMER',"THE CRAWFORD GROUP INC" , "ENTERPRISE RAC")
df = apply_rule_starts_with(df, "RULE 055",'DNB_CUSTOMER_NAME',"STATE OF NEW YORK" , "STATE OF NEW YORK")
df = apply_rule_starts_with(df, "RULE 056",'DNB_CUSTOMER_NAME',"STATE OF GEORGIA" , "STATE OF GEORGIA")
df = apply_rule_starts_with(df, "RULE 057",'DNB_CUSTOMER_NAME',"VERIZON COMMUNICATIONS INC" , "VERIZON SOURCING LLC")
df = apply_rule_starts_with(df, "RULE 058",'DNB_CUSTOMER_NAME',"STATE OF NORTH CAROLINA" , "STATE OF NORTH CAROLINA")
df = apply_rule_starts_with(df, "RULE 059",'CUSTOMER',"DYCOM INDUSTRIES" , "DYCOM INDUSTRIES")
df = apply_rule_starts_with(df, "RULE 060",'CUSTOMER',"TERMINIX" , "TERMINIX CONSUMER SERVICES LLC")
df = apply_rule_starts_with(df, "RULE 061",'CUSTOMER',"WAYNE MUTUAL INSURANCE CO" , "WAYNE MUTUAL INSURANCE CO")

df = apply_rule_starts_with(df, "RULE 062",'CUSTOMER',"ART WATT PAINTING" , "ART WATT PAINTING")
df = apply_rule_starts_with(df, "RULE 063",'CUSTOMER',"BUENA VISTA SECURITY AND PROTE" , "BUENA VISTA SECURITY AND PROTE")
df = apply_rule_starts_with(df, "RULE 064",'CUSTOMER',"GREAT STONE GRANITE" , "GREAT STONE GRANITE")
df = apply_rule_starts_with(df, "RULE 065",'CUSTOMER',"SCHULTHEIS BROS CO" , "SCHULTHEIS BROS CO")
df = apply_rule_starts_with(df, "RULE 066",'CUSTOMER',"PIECE OF GREEN" , "PIECE OF GREEN LANDSCAPING")
df = apply_rule_starts_with(df, "RULE 067",'CUSTOMER',"CLEARPOINT CONSULTING ENGINEER" , "CLEARPOINT CONSULTING ENGINEER")
df = apply_rule_starts_with(df, "RULE 068",'CUSTOMER',"GRADY CRAWFORD CONSTRUCTION CO" , "GRADY CRAWFORD CONSTRUCTION CO")
df = apply_rule_starts_with(df, "RULE 069",'CUSTOMER',"ALLSOUTH APPLIANCE GROUP" , "ALLSOUTH APPLIANCE GROUP")
df = apply_rule_starts_with(df, "RULE 070",'CUSTOMER',"DAC ENTERPRISES" , "DAC ENTERPRISES")
df = apply_rule_starts_with(df, "RULE 071",'CUSTOMER',"HIGHLAND COMMUNITY COLLEGE" , "HIGHLAND COMMUNITY COLLEGE")
df = apply_rule_starts_with(df, "RULE 072",'CUSTOMER',"MATRIX SERVICE COMPANY" , "MATRIX SERVICE COMPANY")
df = apply_rule_starts_with(df, "RULE 073",'CUSTOMER',"TWIN CITY LIMOUSINES" , "TWIN CITY LIMOUSINES")
df = apply_rule_starts_with(df, "RULE 074",'CUSTOMER',"RMI SERVICES CORPORATION" , "RMI SERVICES CORPORATION")
df = apply_rule_starts_with(df, "RULE 075",'CUSTOMER',"GREAT LAKES HOME HEALTH S" , "GREAT LAKES HOME HEALTH S")
df = apply_rule_starts_with(df, "RULE 076",'CUSTOMER',"TOTAL DEPTH" , "TOTAL DEPTH")
df = apply_rule_starts_with(df, "RULE 077",'CUSTOMER',"CAPE BUILDING SYSTEMS" , "CAPE BUILDING SYSTEMS")
df = apply_rule_starts_with(df, "RULE 078",'CUSTOMER',"STATE OF TEXAS" , "STATE OF TEXAS")
df = apply_rule_starts_with(df, "RULE 079",'CUSTOMER',"RICS ELECTRIC" , "RICS ELECTRIC")
df = apply_rule_starts_with(df, "RULE 080",'CUSTOMER',"GILS CARPETS" , "GILS CARPETS")
df = apply_rule_starts_with(df, "RULE 081",'CUSTOMER',"PREMIER THERAPY" , "PREMIER THERAPY")
df = apply_rule_starts_with(df, "RULE 082",'CUSTOMER',"STE GENEVIEVE COUNTY MEMORIAL" , "STE GENEVIEVE COUNTY MEMORIAL")
df = apply_rule_starts_with(df, "RULE 083",'CUSTOMER',"STILSING ELECTRIC INC" , "STILSING ELECTRIC INC")
df = apply_rule_starts_with(df, "RULE 084",'CUSTOMER',"ROBINSON AUTO SALES" , "ROBINSON AUTO SALES")
df = apply_rule_starts_with(df, "RULE 085",'CUSTOMER',"FLORIDA FIRST CALL REMOVAL SER" , "FLORIDA FIRST CALL REMOVAL SER")
df = apply_rule_starts_with(df, "RULE 086",'CUSTOMER',"LANDMARK CONSTRUCTION SOLUTIO" , "LANDMARK CONSTRUCTION SOLUTIO")
df = apply_rule_starts_with(df, "RULE 087",'CUSTOMER',"IMPORT MOTORS OF OLD SAYBROOK" , "IMPORT MOTORS OF OLD SAYBROOK")
df = apply_rule_starts_with(df, "RULE 088",'CUSTOMER',"RALEIGH EAST CONCRETE CONSTRUC" , "RALEIGH EAST CONCRETE CONSTRUC")
df = apply_rule_starts_with(df, "RULE 089",'CUSTOMER',"AMES PLUMBING SERVIC" , "AMES PLUMBING SERVIC")
df = apply_rule_starts_with(df, "RULE 090",'CUSTOMER',"FENIGOR GROUP LLC" , "FENIGOR GROUP LLC")
df = apply_rule_starts_with(df, "RULE 091",'CUSTOMER',"MONTCALM COUNTY ROAD COMMISSIO" , "MONTCALM COUNTY ROAD COMMISSIO")
df = apply_rule_starts_with(df, "RULE 092",'CUSTOMER',"WESTERN INDUSTRIAL CONTRACTORS" , "WESTERN INDUSTRIAL CONTRACTORS")
df = apply_rule_starts_with(df, "RULE 093",'CUSTOMER',"TIRE CORRAL OF AMERICA" , "TIRE CORRAL OF AMERICA")
df = apply_rule_starts_with(df, "RULE 094",'CUSTOMER',"CARVANA" , "CARVANA")

df['EDW_CUSTOMER_NAME'] = df['EDW_CUSTOMER_NAME_ORIGINAL']
df['CUSTOMER_ACCOUNT_NAME'] = df['CUSTOMER_ACCOUNT_NAME_ORIGINAL']

# stupid Tableau can't deal with the addition of a column
# need to figure out how to add this back in
# okay -> turns out this is not down to Tableau, but the stupid way I built the view
del(df['CUST_CALC_RULE'])
del(df['EDW_CUSTOMER_NAME_ORIGINAL'])
del(df['CUSTOMER_ACCOUNT_NAME_ORIGINAL'])

#print(len(df))
#print(len(df.CUSTOMER.unique()))
#df.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_j = pd.merge(df, ACCOUNTS_WITH_EBX_PARTY_df, on='CUSTOMER_ACCOUNT_ID', how='left')
df_j.loc[~df_j["PARTY_DEFAULT_NAME"].isnull(),'CUSTOMER'] = df_j.PARTY_DEFAULT_NAME
df_j.loc[~df_j["PARTY_DEFAULT_NAME"].isnull(),'CUST_CALC_SOURCE'] = 'MDM'
df_j.CUST_CALC_SOURCE.value_counts()

del(df_j['PARTY_ID'])
del(df_j['PARTY_DEFAULT_NAME'])

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
df_j.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
RDW_CONVERSIONS_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_r = RDW_CONVERSIONS_df[['FLEET_ID','CLASSIC_ACCOUNT_NUMBER']].dropna(subset=['CLASSIC_ACCOUNT_NUMBER']).copy()
df_r.FLEET_ID = df_r.FLEET_ID.str.strip()
df_r = df_r[~df_r['FLEET_ID'].str.contains('-',na=False)]
df_r.FLEET_ID = df_r.FLEET_ID.astype('float')
df_r.FLEET_ID = df_r.FLEET_ID.astype('Int64')
print(len(df_r))
print()

df_cust = df_j[['CUSTOMER_ACCOUNT_ID','CUSTOMER']].copy()
df_cust.CUSTOMER_ACCOUNT_ID = df_cust['CUSTOMER_ACCOUNT_ID'].astype('Int64')
print(len(df_cust))
print()

df_rj = pd.merge(df_r, df_cust, left_on='FLEET_ID', right_on='CUSTOMER_ACCOUNT_ID', how='inner')
df_rj = df_rj[pd.to_numeric(df_rj.CLASSIC_ACCOUNT_NUMBER, errors='coerce').notnull()]
df_rj.CLASSIC_ACCOUNT_NUMBER = df_rj.CLASSIC_ACCOUNT_NUMBER.astype(float)
df_rj.CLASSIC_ACCOUNT_NUMBER = df_rj.CLASSIC_ACCOUNT_NUMBER.astype(np.int64)
print(len(df_rj))

df_cust_classic = df_j[['CUSTOMER_ACCOUNT_ID','CUSTOMER']].copy()
df_cust_classic.columns = ['CUSTOMER_ACCOUNT_ID', 'CLASSIC_CUSTOMER']
df_cust_classic.CUSTOMER_ACCOUNT_ID = df_cust['CUSTOMER_ACCOUNT_ID'].astype('Int64')
print(len(df_cust_classic))

df_rj = df_rj[['FLEET_ID','CLASSIC_ACCOUNT_NUMBER','CUSTOMER']]
df_rj = pd.merge(df_rj, df_cust_classic, left_on='CLASSIC_ACCOUNT_NUMBER', right_on='CUSTOMER_ACCOUNT_ID', how='inner')
print(len(df_rj))

df_rdw_conversions = df_rj[df_rj.CUSTOMER!=df_rj.CLASSIC_CUSTOMER]
print(len(df_rdw_conversions))
df_rdw_conversions.head()

df_rdw_conversions = df_rdw_conversions[['CUSTOMER_ACCOUNT_ID','CUSTOMER']]
df_rdw_conversions.columns = ['CUSTOMER_ACCOUNT_ID','CONVERSION_REPLACEMENT_CUSTOMER']
df_rdw_conversions.drop_duplicates(subset=['CUSTOMER_ACCOUNT_ID'], inplace=True)
print(len(df_rdw_conversions))
df_rdw_conversions.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df_j))
df_j_with_rdw_conversions = pd.merge(df_j, df_rdw_conversions, on='CUSTOMER_ACCOUNT_ID',how='left')
print(len(df_j_with_rdw_conversions))

df_j_with_rdw_conversions.loc[~df_j_with_rdw_conversions["CONVERSION_REPLACEMENT_CUSTOMER"].isnull(),'CUSTOMER'] = df_j_with_rdw_conversions.CONVERSION_REPLACEMENT_CUSTOMER
df_j_with_rdw_conversions.loc[~df_j_with_rdw_conversions["CONVERSION_REPLACEMENT_CUSTOMER"].isnull(),'CUST_CALC_SOURCE'] = 'RDW CONVERSIONS'

df_j = df_j_with_rdw_conversions
del(df_j['CONVERSION_REPLACEMENT_CUSTOMER'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_j.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
unique_customer_list = df_j.CUSTOMER.unique()
df_customer_ids = pd.DataFrame(unique_customer_list)
df_customer_ids.columns = ["CUSTOMER"]
df_customer_ids = df_customer_ids.sort_values(['CUSTOMER']).reset_index(drop=True)
df_customer_ids = df_customer_ids.reset_index(drop=False)
df_customer_ids['CUSTOMER_ID'] = df_customer_ids.index + 77000000
del(df_customer_ids['index'])
df_customer_ids.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(len(df_j))
df_jj = pd.merge(df_j, df_customer_ids, on='CUSTOMER')
df_jj.dropna(subset=['CUSTOMER'], inplace=True)
print(len(df_jj))
df_jj.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_by_account = df_jj[['CUSTOMER_ACCOUNT_ID','CUSTOMER_ID', 'CUSTOMER']]
df_by_account.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
len(df_by_account.CUSTOMER_ID.unique())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_jj.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ACCOUNT_NEW_SALES_FULL_df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ACCOUNT_NEW_SALES_FULL_df.columns.tolist()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#print(len(df_jj))
#print(len(ACCOUNT_NEW_SALES_df))
#ACCOUNT_NEW_SALES_df['HAS_SALES_FLAG'] = True
#df_j_with_sales = pd.merge(df_jj, ACCOUNT_NEW_SALES_df, on='CUSTOMER_ACCOUNT_ID', how='left')
#print(len(df_j_with_sales))

print(len(df_jj))
print(len(ACCOUNT_NEW_SALES_FULL_df))
ACCOUNT_NEW_SALES_FULL_df.columns = ['SALES_MARKETING_PARTNER_NM','SALES_BUSINESS_PROGRAM_NM','SALES_PROGRAM_ID','CUSTOMER_ACCOUNT_ID','SALES_CAMPAIGN_TYPE','SALES_COUPON_CODE','SALES_CHANNEL','SALES_REP','SALES_TRANS_RECORDS','SALES_DATA_SOURCE','HAS_SALES_FLAG']
ACCOUNT_NEW_SALES_FULL_df['HAS_SALES_FLAG'] = True
df_j_with_sales = pd.merge(df_jj, ACCOUNT_NEW_SALES_FULL_df, on='CUSTOMER_ACCOUNT_ID', how='left')
print(len(df_j_with_sales))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_j_with_sales.HAS_SALES_FLAG.value_counts(dropna=False)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df = df_j_with_sales

# Write recipe outputs
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS")
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS.write_with_schema(ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df)

BY_ACCOUNT = dataiku.Dataset("BY_ACCOUNT")
BY_ACCOUNT.write_with_schema(df_by_account)
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = ACCOUNTS_WITH_BUNDLER_AND_DUNS_df

import warnings
warnings.filterwarnings(action='once')

df['DUNS'] = df['DUNS'].astype('Int64', errors='ignore')
df['DNB_DUNS_NUMBER'] = df['DNB_DUNS_NUMBER'].astype('Int64', errors='ignore')
df['DNB_BUSINESS_NAME'] = df['DNB_BUSINESS_NAME'].str.upper()
df["DNB_BUSINESS_NAME"] = df['DNB_BUSINESS_NAME'].str.replace('[^\w\s]','')

df['DNB_GLOBAL_ULT_NUMBER'] = df['DNB_GLOBAL_ULT_NUMBER'].astype('Int64', errors='ignore')
df['DNB_GLOBAL_ULT_NAME'] = df['DNB_GLOBAL_ULT_NAME'].str.upper()
df["DNB_GLOBAL_ULT_NAME"] = df['DNB_GLOBAL_ULT_NAME'].str.replace('[^\w\s]','')

df['DNB_DOMESTIC_ULT_NUMBER'] = df['DNB_DOMESTIC_ULT_NUMBER'].astype('Int64', errors='ignore')
df['DNB_DOMESTIC_ULTIMATE_NAME'] = df['DNB_DOMESTIC_ULTIMATE_NAME'].str.upper()
df["DNB_DOMESTIC_ULTIMATE_NAME"] = df['DNB_DOMESTIC_ULTIMATE_NAME'].str.replace('[^\w\s]','')

df['DNB_HQ_NUMBER'] = df['DNB_HQ_NUMBER'].astype('Int64', errors='ignore')
df['DNB_HQ_NAME'] = df['DNB_HQ_NAME'].str.upper()
df["DNB_HQ_NAME"] = df['DNB_HQ_NAME'].str.replace('[^\w\s]','')

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

df.loc[df["CUSTOMER"].isnull(),'CUST_CALC_SOURCE'] = 'ACCOUNT'
df.loc[df["CUSTOMER"].isnull(),'CUSTOMER'] = df["CUSTOMER_ACCOUNT_NAME"]

# RULE SETs
def apply_rule(df, rule_name,filter_name_list,final_name):

    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUST_CALC_SOURCE"] = "Rule (IsIn)"
    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUST_CALC_RULE"] = rule_name
    df.loc[df['CUSTOMER'].isin(filter_name_list),"CUSTOMER"] = final_name

    return(df)

def apply_rule_starts_with(df, rule_name, compares_to, starts_with_string,final_name):

    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUST_CALC_SOURCE"] = "RULE (Startswith)"
    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUST_CALC_RULE"] = rule_name
    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUSTOMER"] = final_name

    return(df)

def apply_rule_contains(df, rule_name, compares_to, contains_string,final_name):

    df.loc[df[compares_to].str.contains(contains_string, na=False),"CUST_CALC_SOURCE"] = "RULE (Contains)"
    df.loc[df[compares_to].str.contains(contains_string, na=False),"CUST_CALC_RULE"] = rule_name
    df.loc[df[compares_to].str.contains(contains_string, na=False),"CUSTOMER"] = final_name

    return(df)

# this set of rules represents high card count mappings
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

df['EDW_CUSTOMER_NAME'] = df['EDW_CUSTOMER_NAME_ORIGINAL']
df['CUSTOMER_ACCOUNT_NAME'] = df['CUSTOMER_ACCOUNT_NAME_ORIGINAL']

# stupid Tableau can't deal with the addition of a column
# need to figure out how to add this back in
del(df['CUST_CALC_RULE'])
del(df['EDW_CUSTOMER_NAME_ORIGINAL'])
del(df['CUSTOMER_ACCOUNT_NAME_ORIGINAL'])

#print(len(df))
#print(len(df.CUSTOMER.unique()))
#df.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_j = pd.merge(df, ACCOUNTS_WITH_EBX_PARTY_df, on='CUSTOMER_ACCOUNT_ID', how='left')
df_j.loc[~df_j["PARTY_DEFAULT_NAME"].isnull(),'CUSTOMER'] = df_j.PARTY_DEFAULT_NAME
df_j.loc[~df_j["PARTY_DEFAULT_NAME"].isnull(),'CUST_CALC_SOURCE'] = 'EBX'
df_j.CUST_CALC_SOURCE.value_counts()

del(df_j['PARTY_ID'])
del(df_j['PARTY_DEFAULT_NAME'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df = df_j

# Write recipe outputs
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS")
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS.write_with_schema(ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df)

#BY_ACCOUNT = dataiku.Dataset("BY_ACCOUNT")
#BY_ACCOUNT.write_with_schema(BY_ACCOUNT_df)

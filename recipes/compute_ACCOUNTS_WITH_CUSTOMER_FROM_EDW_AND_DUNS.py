# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import string

# Read recipe inputs
ACCOUNTS_WITH_BUNDLER_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_BUNDLER_AND_DUNS")
ACCOUNTS_WITH_BUNDLER_AND_DUNS_df = ACCOUNTS_WITH_BUNDLER_AND_DUNS.get_dataframe()

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

# this set of rules represents high card count mappings
# these rules have been manually verified
# as they impact large numbers of cards (and in turn gallons, spend and revenue)
df = apply_rule(df, "RULE 001", ['QUANTA SERVICES INC','QUANTA SERVICES'], 'QUANTA SERVICES INC')
df = apply_rule(df, "RULE 002", ['0113 WINDSTREAM COMM','0113 WINDSTREAM COMM (2)'], '0113 WINDSTREAM COMM')
df = apply_rule(df, "RULE 003", ['1033 MONSANTO COMPANY (25)','1033 MONSANTO COMPANY (32)'], '1033 MONSANTO COMPANY')
df = apply_rule(df, "RULE 004", ['2536 HOME DEPOT','2536 HOME DEPOT 5'], '2536 HOME DEPOT')
df = apply_rule(df, "RULE 005", ['3274 MEDTRONIC 2','3274 MEDTRONIC','3274 MEDTRONIC AD'], '3274 MEDTRONIC')
df = apply_rule(df, "RULE 006", ['3373 BASF','3373 BASF AD'], '3373 BASF')
df = apply_rule(df, "RULE 007", ['5929-TESLA (2)','5929-TESLA','5929-TESLA (3)'], '5929-TESLA')

df = apply_rule(df, "RULE 008", ['6220-KONE INC','6220-KONE INC','6220-KONE INC (3)'], '6220-KONE INC')
df = apply_rule(df, "RULE 009", ['7325 ADVANCE STORES COMPANY 4','7325 ADVANCE AUTO','7325 ADVANCE STORES COMP','7325 ADVANCE STORES COMP 2'], 'ADVANCE AUTO')
df = apply_rule(df, "RULE 010", ['17435-GE HEALTHCARE (3)','17435 GE HEALTHCARE','17435-GE HEALTHCARE','17435-GE HEALTHCARE (2)'], '17435-GE HEALTHCARE')
df = apply_rule(df, "RULE 011", ['17595-SCHLUMBERGER','17595-SCHLUMBERGER','17595-SCHLUMBERGER (2)','17595 SCHLUMBERGER AD'], '17595 SCHLUMBERGER')
df = apply_rule(df, "RULE 012", ['ADAPTHEALTH CORP','ADAPTHEALTH LLC'], 'ADAPTHEALTH')
df = apply_rule(df, "RULE 013", ['AGILENT TECHNOLOGIES LP','AGILENT TECHNOLOGIES LP 2'], 'AGILENT TECHNOLOGIES LP')
df = apply_rule(df, "RULE 014", ['AGL RESOURCES 5CU6','AGL RESOURCES ARI'], 'AGL RESOURCES')
df = apply_rule(df, "RULE 015", ['APTIVE ENVIRONMENTAL LLC','ADAPTIVE ENVIRONMENTAL CONSULTING INC','APTIVE ENVIRONMENTAL LLC','APTIVE ENVIRONMENTAL 0CV7'], 'APTIVE ENVIRONMENTAL')
df = apply_rule(df, "RULE 016", ['ARAMARK (RPT PARENT)','ARAMARK RPT PARENT','ARAMARK FOOD & FACILITY HYBRID','ARAMARK','ARAMARK 2G93'], 'ARAMARK')
df = apply_rule(df, "RULE 017", ['CABLE ONE (5R09)','CABLE ONE INC 5R09','CABLE ONE INC'], 'CABLE ONE INC')
df = apply_rule(df, "RULE 018", ['CBRE 3','CBRE','CBRE 2','CBRE GROUP INC'], 'CBRE GROUP INC')
df = apply_rule(df, "RULE 019", ['COMCAST CABLE','COMCAST CABLE HQ 5BA6','COMCAST CABLE WEST 5BA6','COMCAST CABLE NE 5BA6','COMCAST CABLE CE 5BA6','COMCAST CABLE COMMUN 5FH6'], 'COMCAST CABLE')
df = apply_rule(df, "RULE 020", ['COMPASS GROUP','COMPASS GROUP PLC','COMPASS GROUP 5968'], 'COMPASS GROUP')
df = apply_rule(df, "RULE 021", ['CONOCOPHILLIPS','CONOCOPHILLIPS COMPANY','CONOCOPHILLIPS CO'], 'CONOCOPHILLIPS')
df = apply_rule(df, "RULE 022", [' MMCCARTHY TIRE','MCCARTHY TIRE'], 'MCCARTHY TIRE')

df = apply_rule(df, "RULE 023", ['',''], '')
df = apply_rule(df, "RULE 024", ['CSC SERVICEWORKS','CSC SERVICEWORKS 0R53'], 'CSC SERVICEWORKS')
df = apply_rule(df, "RULE 025", ['CROWN CASTLE USA (0AK7)','CROWN CASTLE USA INC 0AK7'], 'CROWN CASTLE USA INC')
df = apply_rule(df, "RULE 026", ['E JOHNSON CONTROLS SECURITY','E JOHNSON CONTROLS FIRE & SEC'], 'E JOHNSON CONTROLS FIRE & SEC')
df = apply_rule(df, "RULE 027", ['FASTENAL (0469)(3)','FASTENAL (0469)(4)','FASTENAL (0469)','FASTENAL (0469)(2)','FASTENAL COMPANY'], 'FASTENAL COMPANY')
df = apply_rule(df, "RULE 028", ['FOSS NATIONAL/CORP-RATEFOSS NATIONAL LEASING (2)','FOSS NATIONAL LEASING (2)'], 'FOSS NATIONAL LEASING')
df = apply_rule(df, "RULE 029", ['GENERAL MILLS','GENERAL MILLS (3GM0)','GENERAL MILLS SALES','GENERAL MILLS OPERATIONS  3GMO'], 'GENERAL MILLS')
df = apply_rule(df, "RULE 030", ['HELMERICH AND PAYNE INC PARENT','HELMERICH  PAYNE ID 5FM8'], 'HELMERICH AND PAYNE INC')
df = apply_rule(df, "RULE 031", ['HUSSMANN CORPORATION','HUSSMAN CORPORATION'],'HUSSMAN CORPORATION')
df = apply_rule(df, "RULE 032", ['IGT GLOBAL SOLUTIONS CORP','IGT GLOBAL SOLUTIONS 0DK1'], '')
df = apply_rule(df, "RULE 033", ['J R SIMPLOT -S40','J R SIMPLOT -S40 (2)','J R SIMPLOT CO'], 'J R SIMPLOT CO')
df = apply_rule(df, "RULE 034", ['JC EHRLICH (4)','JC EHRLICH (5)','JC EHRLICH (2)'], 'JC EHRLICH')
df = apply_rule(df, "RULE 035", ['JOHNSONJOHNSON','JOHNSON JOHNSON CITRUS'], 'JOHNSON JOHNSON')
df = apply_rule(df, "RULE 036", ['KINDER MORGAN (0469)(3)','KINDER MORGAN INC','KINDER MORGAN 5GU5','KINDER MORGAN INC'], 'KINDER MORGAN INC')

df = apply_rule(df, "RULE 037", ["L'OREAL USA 3","L'OREAL USA","L'OREAL USA 2","L'OREAL CANADA INC."], "L'OREAL USA")
df = apply_rule(df, "RULE 038", ['LABCORP','LABCORP (3LAB)','LABCORP (3LAB)(2)','LABORATORY CORPORATION OF AMERICA'], 'LABORATORY CORPORATION OF AMERICA')
df = apply_rule(df, "RULE 039", ['LEHIGH HANSON (3LHN)','LEHIGH HANSON (3LHN)(2)','3995 LEHIGH HANSON INC','116710-LEHIGH HANSON, INC (2)',
                                 '116710-LEHIGH HANSON, INC (3)','116710-LEHIGH HANSON, INC (4)','116710-LEHIGH HANSON, INC (6)',
                                 "E450 LEHIGH HANSON MATERIALS L",'LEHIGH HANSON INC3LHC'], 'LEHIGH HANSON INC')
df = apply_rule(df, "RULE 040", ['LIBERTY MUTUAL 2D93','LIBERTY MUTUAL GROUP'], 'LIBERTY MUTUAL GROUP')
df = apply_rule(df, "RULE 041", ['MARATHON PETROLEUM CORPORATION','MARATHON PETROLEUM COMPANY LP','MARATHON PETROLEUMEQUIP'], 'MARATHON PETROLEUM')
df = apply_rule(df, "RULE 042", ['MONDELEZ GLOBAL LLC','MONDELEZ GLOBAL LLC 2'], 'MONDELEZ GLOBAL LLC')
df = apply_rule(df, "RULE 043", ['NATIONAL FUEL (2G35)','NATIONAL FUEL GAS 5AP6'], 'NATIONAL FUEL')

df = apply_rule(df, "RULE 044", ['NATIONAL FUEL GAS 5AP6','NATIONAL FUEL AND LUBRICANT IN','NATIONAL FUEL (2G35)','NATIONAL FUEL'], 'NATIONAL FUEL')
df = apply_rule(df, "RULE 045", ['NEXSTAR BROADCASTING (5R10)','NEXSTAR BROADCASTING 5R10'], 'NEXSTAR BROADCASTING')

df = apply_rule(df, "RULE 046", ['NORFOLK SOUTHERN RAILWAY CO (3','NORFOLK SOUTHERN 5K16','NORFOLK SOUTHERN 5K162'], 'NORFOLK SOUTHERN')
df = apply_rule(df, "RULE 047", ['NORTHERN CLEARING INC 3','NORTHERN CLEARING INC 4'], 'NORTHERN CLEARING INC')
df = apply_rule(df, "RULE 048", ['NOVONORDISK','NOVO NORDISK','NOVO NORDISK INC','NOVO NORDISK FONDEN'], 'NOVO NORDISK INC')
df = apply_rule(df, "RULE 049", ['PHILLIPS 66 COMPANY (2J56)','PHILLIPS 66 COMPANY 2J56'], 'PHILLIPS 66 COMPANY')
df = apply_rule(df, "RULE 050", ['SCHINDLER ELEVATOR (69)','SCHINDLER ELEVATOR 5G70'], 'SCHINDLER ELEVATOR')
df = apply_rule(df, "RULE 051", ['SCI MANAGEMENT','SCI MANAGEMENT (2)'], 'SCI MANAGEMENT')
df = apply_rule(df, "RULE 052", ['SERVICE EXPERTS','SERVICE EXPERTS LD'], 'SERVICE EXPERTS')
df = apply_rule(df, "RULE 053", ['STONEMOR INC 0DS4','STONEMOR PARTNERS'], 'STONEMOR Inc')
df = apply_rule(df, "RULE 054", ['SYNGENTA (WHEELS)(2)','SYNGENTA (WHEELS)','SYNGENTA US HOLDING INC'], 'SYNGENTA')
df = apply_rule(df, "RULE 055", ['TRANSDEV (0R64)','TRANSDEV NORTH AMERI 0R64'], 'TRANSDEV')
df = apply_rule(df, "RULE 056", ['UNITED RENTALS (ARI)','UNITED RENTALS 5T55','UNITED RENTALS INC'], 'UNITED RENTALS INC')
df = apply_rule(df, "RULE 057", ['US LBM (5EE8)','US LBM (5EE8)(2)','US LBM HOLDINGS, LLC (5EE8)','LAMPERT YARDS US LBM LLC','US LBM HOLDINGS LLC 5EE8'], 'US LBM HOLDINGS LLC')
df = apply_rule(df, "RULE 058", ['VAN POOL TRANSPORTATION LLC','VAN POOL TRANSPORTATION LLC 3','VAN POOL TRANSPORTATION LLC 4','VAN POOL TRANSPORTATION'], 'VAN POOL TRANSPORTATION')
df = apply_rule(df, "RULE 059", ['VEOLIA WATER LOGISTICS (2R63)','VEOLIA LOGISTICS 2R63'], 'VEOLIA LOGISTICS')
df = apply_rule(df, "RULE 060", ['WILLIAMS STRATEGIC (0AX6)','WILLIAMS STRATEGIC (0AX6)(2)','WILLIAMS STRATEGIC 5DB0'], 'WILLIAMS STRATEGIC')
df = apply_rule(df, "RULE 061", ['XTO ENERGY (2M33)','XTO ENERGY (2M33)(1)','XTO ENERGY (2M33)','XTO ENERGY CANADA','XTO ENERGY INC 2M33'], 'XTO ENERGY INC')

def apply_rule_starts_with(df, rule_name, compares_to, starts_with_string,final_name):

    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUST_CALC_SOURCE"] = "RULE (Startswith)"
    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUST_CALC_RULE"] = rule_name
    df.loc[df[compares_to].str.startswith(starts_with_string, na=False),"CUSTOMER"] = final_name

    return(df)

df = apply_rule_starts_with(df, "RULE 062",'CUSTOMER',"MANSFIELD OIL" , "MANSFIELD OIL")
df = apply_rule_starts_with(df, "RULE 063",'CUSTOMER',"ASPLUNDH" , "ASPLUNDH")
df = apply_rule_starts_with(df, "RULE 064",'CUSTOMER',"CINTAS CORPORATION" , "CINTAS CORPORATION")
df = apply_rule_starts_with(df, "RULE 065",'CUSTOMER',"ENTERPRISE RAC" , "ENTERPRISE RAC")
df = apply_rule_starts_with(df, "RULE 066",'CUSTOMER',"THE CRAWFORD GROUP INC" , "THE CRAWFORD GROUP INC")
df = apply_rule_starts_with(df, "RULE 067",'DNB_CUSTOMER_NAME',"STATE OF NEW YORK" , "STATE OF NEW YORK")
df = apply_rule_starts_with(df, "RULE 068",'DNB_CUSTOMER_NAME',"STATE OF GEORGIA" , "STATE OF GEORGIA")
df = apply_rule_starts_with(df, "RULE 069",'DNB_CUSTOMER_NAME',"VERIZON COMMUNICATIONS INC" , "VERIZON SOURCING LLC")
df = apply_rule_starts_with(df, "RULE 070",'DNB_CUSTOMER_NAME',"STATE OF NORTH CAROLINA" , "STATE OF NORTH CAROLINA")
df = apply_rule_starts_with(df, "RULE 071",'CUSTOMER',"DYCOM INDUSTRIES" , "DYCOM INDUSTRIES")
df = apply_rule_starts_with(df, "RULE 072",'CUSTOMER',"TERMINIX" , "TERMINIX CONSUMER SERVICES LLC")

def apply_rule_contains(df, rule_name, compares_to, contains_string,final_name):

    df.loc[df[compares_to].str.startswith(contains_string, na=False),"CUST_CALC_SOURCE"] = "RULE (Contains)"
    df.loc[df[compares_to].str.startswith(contains_string, na=False),"CUST_CALC_RULE"] = rule_name
    df.loc[df[compares_to].str.startswith(contains_string, na=False),"CUSTOMER"] = final_name

    return(df)

df = apply_rule_starts_with(df, "RULE 073",'CUSTOMER',"BIMBO" , "BIMBO BAKERIES USA INC")

del(df['CUST_CALC_RULE'])

#print(len(df))
#df.CUST_CALC_SOURCE.value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#df[df.DNB_CUSTOMER_NAME=='STATE OF NEW YORK'].head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df = df

# Write recipe outputs
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS = dataiku.Dataset("ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS")
ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS.write_with_schema(ACCOUNTS_WITH_CUSTOMER_FROM_EDW_AND_DUNS_df)
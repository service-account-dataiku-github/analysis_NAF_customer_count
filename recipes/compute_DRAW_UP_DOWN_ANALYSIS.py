# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
CALCULATED_CARD_DRAW_UPS_FULL = dataiku.Dataset("CALCULATED_CARD_DRAW_UPS_FULL")
CALCULATED_CARD_DRAW_UPS_FULL_df = CALCULATED_CARD_DRAW_UPS_FULL.get_dataframe()

CALCULATED_CARD_DRAW_DOWNS_FULL = dataiku.Dataset("CALCULATED_CARD_DRAW_DOWNS_FULL")
CALCULATED_CARD_DRAW_DOWNS_FULL_df = CALCULATED_CARD_DRAW_DOWNS_FULL.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_down = CALCULATED_CARD_DRAW_DOWNS_FULL_df
df_up = CALCULATED_CARD_DRAW_UPS_FULL_df

print(len(df_down))
print(len(df_up))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import time
import difflib
from fuzzywuzzy import fuzz

customers_ = ['AMGEN USA INC', 'OMEROS CORP PO#100752', 'BRITE LINE ASPHALT MAINTENANCE', 'MOSS FARMS (04)(2)',
              'JAMES H COWAN & ASSOC INC','WATTS EQUIPMENT CO INC','CUIVRE RIVER ELECT','BENTONS EQUIPMENT & CONSTRUCTI',
             'MILLENNIUM PHARMA','CONSTELLATION BRANDS (3CRW)']

_cut_off = [0.95, 0.90, 0.8, 0.7, 0.6]

for c in customers_:

    print('processing:', c)
    
    t0 = time.time()
    for co in _cut_off:

        print("cut off", co)
        matches = difflib.get_close_matches(c, df_up['CUSTOMER'].unique(), n=300, cutoff=co)

        if len(matches)>0:
            t1 = time.time()
            avg_duration = ((t1-t0))
            print(len(matches), "matches", avg_duration)
            print()
            print(matches)
        else:
            print("no matches", avg_duration)
        print()
        
    print()
    print("---")
    print()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

#DRAW_UP_DOWN_ANALYSIS_df = ... # Compute a Pandas dataframe to write into DRAW_UP_DOWN_ANALYSIS

# Write recipe outputs
#DRAW_UP_DOWN_ANALYSIS = dataiku.Dataset("DRAW_UP_DOWN_ANALYSIS")
#DRAW_UP_DOWN_ANALYSIS.write_with_schema(DRAW_UP_DOWN_ANALYSIS_df)
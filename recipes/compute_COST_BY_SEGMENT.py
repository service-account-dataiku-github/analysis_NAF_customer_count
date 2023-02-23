# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

from datetime import date, datetime, timedelta
import time

import matplotlib.pyplot as plt

t0 = time.time()

# Read recipe inputs
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019 = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019")
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019_df = NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2019.get_dataframe()

NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021 = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021")
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021_df = NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2021.get_dataframe()

NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020 = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020")
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020_df = NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2020.get_dataframe()

NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022 = dataiku.Dataset("NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022")
NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022_df = NAFCUSTOMER_REVENUE_AGGREGATED_MRU_2022.get_dataframe()


print('loaded annual card count')

t1 = time.time()
print("load duration", (t1-t0)/60.0, "minutes")

# Compute recipe outputs
# TODO: Write here your actual code that computes the outputs
# NB: DSS supports several kinds of APIs for reading and writing data. Please see doc.

COST_BY_SEGMENT_df = ... # Compute a Pandas dataframe to write into COST_BY_SEGMENT


# Write recipe outputs
COST_BY_SEGMENT = dataiku.Dataset("COST_BY_SEGMENT")
COST_BY_SEGMENT.write_with_schema(COST_BY_SEGMENT_df)
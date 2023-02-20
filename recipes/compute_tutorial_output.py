# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
vgsales = dataiku.Dataset("vgsales")
vgsales_df = vgsales.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

tutorial_output_df = vgsales_df # For this sample code, simply copy input to output


# Write recipe outputs
tutorial_output = dataiku.Dataset("tutorial_output")
tutorial_output.write_with_schema(tutorial_output_df)

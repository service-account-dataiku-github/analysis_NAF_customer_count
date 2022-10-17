# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
BY_ACCOUNT = dataiku.Dataset("BY_ACCOUNT")
BY_ACCOUNT_df = BY_ACCOUNT.get_dataframe()

df = BY_ACCOUNT_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
customers = df.CUSTOMER.unique()
len(customers)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import string


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
class Common_Word:

    def __init__(self, word):

        self.word = word
        self.count = 1

class Common_Word_List:

    def __init__(self):

        self.values = []


    def add_word(self, word):

        found = False
        for w in self.values:
            if w.word==word:
                w.count+=1
                found = True

        if not found:
            self.values.append(Common_Word(word))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
list_ = Common_Word_List()
idx = 0

for c in customers:

    c_str = c.translate(str.maketrans('', '', string.punctuation))
    f = c_str.split()

    for w in f:
        if (len(w)>1) and (not w.isnumeric()):
            list_.add_word(w)
    idx+=1

    if (idx % 10000 == 0):
        print(idx, len(list_.values))

    #if idx>5000:
    #    break;

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
_words = []
_counts = []

_keep_threshold = 100

for w in list_.values:
    if w.count>_keep_threshold:
        _words.append(w.word)
        _counts.append(w.count)

df_words = pd.DataFrame(_words, columns=['WORD'])
df_words['COUNTS'] = _counts

df_words.sort_values(by='COUNTS', ascending=False, inplace=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

COMMON_WORDS_IN_NAMES_df = df_words

# Write recipe outputs
COMMON_WORDS_IN_NAMES = dataiku.Dataset("COMMON_WORDS_IN_NAMES")
COMMON_WORDS_IN_NAMES.write_with_schema(COMMON_WORDS_IN_NAMES_df)
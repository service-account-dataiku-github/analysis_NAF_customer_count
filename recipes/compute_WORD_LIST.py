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
import nltk
from nltk.tag import pos_tag

df_dict = pd.DataFrame(nltk.corpus.words.words(), columns=['word'])
df_dict['word'] = df_dict['word'].str.upper()

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
        

words = ['APPLE', 'PEAR', 'APPLE', 'ORANGE', 'PEAR']
        


for w in words:
    
    
    
for w in list_.values:
    print(w.word, w.count)

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
    
    if (idx % 1000 == 0):
        print(idx, len(list_.values))

    if idx>100000:
        break;
        
        
print()
        
for w in list_.values:
    if w.count>20:
        print(w.word, w.count)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
for w in list_.values:
    if w.count>20:
        print(w.word, w.count)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

WORD_LIST_df = BY_ACCOUNT_df # For this sample code, simply copy input to output


# Write recipe outputs
WORD_LIST = dataiku.Dataset("WORD_LIST")
WORD_LIST.write_with_schema(WORD_LIST_df)
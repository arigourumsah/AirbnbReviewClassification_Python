import csv
import re
import pandas as pd
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import *
pd.options.mode.chained_assignment = None  # default='warn'
import sys
from sklearn.model_selection import KFold

filename = sys.argv[1]
df = pd.read_excel(filename)
df['Polarity'] = df['Polarity'].map({'Negative': 0, 'Positive': 1, 'Neutral': 2})
df = df.sample(frac = 1, random_state = 0).reset_index(drop = True)

def fold_i_of_k(df, i, k):
    n = len(df)
    return df[n*(i-1)//k:n*i//k]

testdata1 = fold_i_of_k(df,1,5)
traindata1 = pd.concat([(fold_i_of_k(df,2,5)), (fold_i_of_k(df,3,5)), (fold_i_of_k(df,4,5)), (fold_i_of_k(df,5,5))], ignore_index = True)

testdata2 = fold_i_of_k(df,2,5)
traindata2 = pd.concat([(fold_i_of_k(df,1,5)), (fold_i_of_k(df,3,5)), (fold_i_of_k(df,4,5)), (fold_i_of_k(df,5,5))], ignore_index = True)

testdata3 = fold_i_of_k(df,3,5)
traindata3 = pd.concat([(fold_i_of_k(df,2,5)), (fold_i_of_k(df,1,5)), (fold_i_of_k(df,4,5)), (fold_i_of_k(df,5,5))], ignore_index = True)

testdata4 = fold_i_of_k(df,4,5)
traindata4 = pd.concat([(fold_i_of_k(df,2,5)), (fold_i_of_k(df,3,5)), (fold_i_of_k(df,1,5)), (fold_i_of_k(df,5,5))], ignore_index = True)

testdata5 = fold_i_of_k(df,5,5)
traindata5 = pd.concat([(fold_i_of_k(df,2,5)), (fold_i_of_k(df,3,5)), (fold_i_of_k(df,4,5)), (fold_i_of_k(df,1,5))], ignore_index = True)

export_excel = traindata1.to_excel(r'cv_train1.xlsx')
export_excel = testdata1.to_excel(r'cv_test1.xlsx')

export_excel = traindata2.to_excel(r'cv_train2.xlsx')
export_excel = testdata2.to_excel(r'cv_test2.xlsx')

export_excel = traindata3.to_excel(r'cv_train3.xlsx')
export_excel = testdata3.to_excel(r'cv_test3.xlsx')

export_excel = traindata4.to_excel(r'cv_train4.xlsx')
export_excel = testdata4.to_excel(r'cv_test4.xlsx')

export_excel = traindata5.to_excel(r'cv_train5.xlsx')
export_excel = testdata5.to_excel(r'cv_test5.xlsx')
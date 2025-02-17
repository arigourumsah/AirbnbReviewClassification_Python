from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pandas_ml import ConfusionMatrix

import pandas as pd
import numpy as np
import nltk
import contractions
import string
import re
import sys
import pickle

filename = sys.argv[1]
testdata=pd.read_excel(filename)

def process_message(message, punctuation = True, lower_case = True, contract = True, tokenize = True, stem = True, stop_words = True):
    if punctuation:
        for punctuation in string.punctuation:
            message = message.replace(punctuation, '')
    if lower_case:
        message = message.lower()
    if contract:
        message = contractions.fix(message)
    if tokenize:
        words = word_tokenize(message)
    if stop_words:
        to_remove = []
        with open('negstopwords.txt', 'r') as filehandle:  
            to_remove = [current_place.rstrip() for current_place in filehandle.readlines()]
        sw = set(stopwords.words('english')).difference(to_remove)
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words
	
testdata['Review'] = testdata['Review'].apply(process_message)
testdata.to_excel("Preprocessed_Testdata.xlsx")
with open('wprob_pos.pickle', 'rb') as handle:
    wprob_pos = pickle.load(handle)
with open('wprob_neg.pickle', 'rb') as handle:
    wprob_neg = pickle.load(handle)
with open('wprob_neu.pickle', 'rb') as handle:
    wprob_neu = pickle.load(handle)
with open('prob_review.pickle', 'rb') as handle:
    prob_review = pickle.load(handle)

prob_pos_review = prob_review['prob_pos_review']
prob_neg_review = prob_review['prob_neg_review']
prob_neu_review = prob_review['prob_neu_review']

def NBpredict(testdata, wprob_pos, wprob_neg, wprob_neu, prob_pos_review, prob_neg_review, prob_neu_review):
    noOfMessages = testdata.shape[0]
    review = testdata['Review']
    result = []
    resultpos = []
    resultneg = []
    resultneu = []
    for i in range (0, noOfMessages):
        p_pos = 0
        p_neg = 0
        p_neu = 0
        for word in review[i]:
            if word in wprob_pos:
                p_pos += np.log10(wprob_pos[word])
                p_neg += np.log10(wprob_neg[word])
                p_neu += np.log10(wprob_neu[word])
        p_pos += np.log10(prob_pos_review)
        p_neg += np.log10(prob_neg_review)
        p_neu += np.log10(prob_neu_review)
        resultpos.append(p_pos)
        resultneg.append(p_neg)
        resultneu.append(p_neu)
        if p_pos>p_neg and p_pos>p_neu:
            result.append(1)
        elif p_neg>p_pos and p_neg>p_neu:
            result.append(0)
        elif p_neu>p_pos and p_neu>p_neg:
            result.append(2)
    return result, resultpos, resultneg, resultneu
	
predictions, resultpos, resultneg, resultneu = NBpredict(testdata, wprob_pos, wprob_neg, wprob_neu, prob_pos_review, prob_neg_review, prob_neu_review)
testdata['Prob Pos'] = resultpos
testdata['Prob Neg'] = resultneg
testdata['Prob Neu'] = resultneu
testdata['Predictions'] = predictions
testdata.to_excel("Classification_Output.xlsx")

#computing tp, fp, tn, and fn
def confusionmatrix(testdata):
    y_true = testdata['Polarity']
    y_pred = testdata['Predictions']
    
    confusion_matrix = ConfusionMatrix(y_true, y_pred)
    confusion_matrix.print_stats()
    
confusionmatrix(testdata)
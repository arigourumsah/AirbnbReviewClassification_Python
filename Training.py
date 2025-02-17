from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import pandas as pd
import numpy as np
import nltk
import contractions
import string
import re
import pickle
import sys
pd.options.mode.chained_assignment = None # Default='warn'

filename = sys.argv[1]
traindata=pd.read_excel(filename)

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
        #sw = stopwords.words('english')
        #to_remove = ['no', 'not', 'only']
        to_remove = []
        # open file and read the content in a list
        with open('negstopwords.txt', 'r') as filehandle:  
            to_remove = [current_place.rstrip() for current_place in filehandle.readlines()]
        sw = set(stopwords.words('english')).difference(to_remove)
        #sw = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    return words
	
traindata['Review'] = traindata['Review'].apply(process_message)
traindata.to_excel("Preprocessed_Traindata.xlsx")
def computeReviewTFDict(data):
    """ Returns a tf dictionary for each review whose keys are all 
    the unique words in the review and whose values are their 
    corresponding tf."""
    #Counts the number of times the word appears in review
    traindata['reviewtf'] = pd.DataFrame(columns=['reviewtf'])
    tfDict = traindata['reviewtf']
    noOfMessages = data.shape[0]
    for i in range(0, noOfMessages):
        review=data[i]
        reviewTFDict = {}
        for word in review:
            if word in reviewTFDict:
                reviewTFDict[word] += 1
            else:
                reviewTFDict[word] = 1
        #Computes tf for each word           
        for word in reviewTFDict:
            reviewTFDict[word] = reviewTFDict[word] / len(review)
        tfDict[i] = reviewTFDict
    return tfDict
	
tfDict = computeReviewTFDict(traindata['Review'])

def computeCountDict():
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears."""
    noOfMessages = tfDict.shape[0]
    countDict = {}
    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    for i in range (0, noOfMessages):
        review = tfDict[i]
        for word in review:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict

countDict = computeCountDict()

def computeIDFDict():
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf."""
    data = traindata['Review']
    noOfMessages = data.shape[0]
    idfDict = {}
    for word in countDict:
        idfDict[word] = np.log10(noOfMessages / countDict[word])
    return idfDict

idfDict = computeIDFDict()

def computeReviewTFIDFDict(tfDict):
    """ Returns a dictionary whose keys are all the unique words in the
    review and whose values are their corresponding tfidf."""
    traindata['reviewtfidf'] = pd.DataFrame(columns=['reviewtfidf'])
    tfidfDict = traindata['reviewtfidf']
    noOfMessages = tfDict.shape[0]
    for i in range(0, noOfMessages):
        review=tfDict[i]
        reviewTFIDFDict = {}
        #For each word in the review, we multiply its tf and its idf.
        for word in review:
            reviewTFIDFDict[word] = review[word] * idfDict[word]
        tfidfDict[i] = reviewTFIDFDict
    return tfidfDict

tfidfDict = computeReviewTFIDFDict(tfDict)

def calc_sumtfidf(traindata):
    sumtfidf_pos = 0
    sumtfidf_neg = 0
    sumtfidf_neu = 0
    tfidf = traindata['reviewtfidf']
    noOfMessages = traindata.shape[0]
    for i in range(0, noOfMessages):
        rev_tfidf = tfidf[i]
        if traindata['Polarity'][i] == 1:
            for word in rev_tfidf:
                sumtfidf_pos += rev_tfidf[word]
        elif traindata['Polarity'][i] == 0:
            for word in rev_tfidf:
                sumtfidf_neg += rev_tfidf[word]
        elif traindata['Polarity'][i] == 2:
            for word in rev_tfidf:
                sumtfidf_neu += rev_tfidf[word]
    return sumtfidf_pos, sumtfidf_neg, sumtfidf_neu
	
#calculate sum of the tfidf values for all the words in each class accordingly
sumtfidf_pos, sumtfidf_neg, sumtfidf_neu = calc_sumtfidf(traindata)

def NBtrain(traindata, sum_tfidf_pos, sum_tfidf_neg, sum_tfidf_neu):
    noOfMessages = traindata.shape[0]
    review = traindata['Review']
    tfidf = traindata['reviewtfidf']
    polarity = traindata['Polarity']
    count_pos_review = 0
    count_neg_review = 0
    count_neu_review = 0
    wordcount_pos = dict()
    wordcount_neg = dict()
    wordcount_neu = dict()
    wprob_pos = dict()
    wprob_neg = dict()
    wprob_neu = dict()
    wordDict = []
    # Run through each review's tf dictionary and increment countDict's (word, doc) pair
    for i in range (0, noOfMessages):
        for word in review[i]:
            if word not in wordDict:
                wordDict.append(word)
                wordcount_pos[word] = 0
                wordcount_neg[word] = 0
                wordcount_neu[word] = 0
    for i in range (0, noOfMessages):
        rowtfidf=tfidf[i]
        if polarity[i] == 1:
            count_pos_review += 1
            for word in review[i]:
                if word in wordDict:
                    wordcount_pos[word] += rowtfidf[word]
        elif polarity[i] == 0:
            count_neg_review += 1
            for word in review[i]:
                if word in wordDict:
                    wordcount_neg[word] += rowtfidf[word]
        elif polarity[i] == 2:
            count_neu_review += 1
            for word in review[i]:
                if word in wordDict:
                    wordcount_neu[word] += rowtfidf[word]
    for dict_word in wordDict:
        wprob_pos[dict_word] = (wordcount_pos[dict_word] + 1) / (sum_tfidf_pos + len(wordDict))
        wprob_neg[dict_word] = (wordcount_neg[dict_word] + 1) / (sum_tfidf_neg + len(wordDict))
        wprob_neu[dict_word] = (wordcount_neu[dict_word] + 1) / (sum_tfidf_neu + len(wordDict))
    prob_pos_review = count_pos_review / (count_pos_review + count_neg_review + count_neu_review)
    prob_neg_review = count_neg_review / (count_pos_review + count_neg_review + count_neu_review)
    prob_neu_review = count_neu_review / (count_pos_review + count_neg_review + count_neu_review)
    return wprob_pos, wprob_neg, wprob_neu, prob_pos_review, prob_neg_review, prob_neu_review

wprob_pos, wprob_neg, wprob_neu, prob_pos_review, prob_neg_review, prob_neu_review = NBtrain(traindata, sumtfidf_pos, sumtfidf_neg, sumtfidf_neu)

print(prob_neu_review)
print(prob_neg_review)

traindata.to_excel("TFIDF_Output.xlsx")
"""with open('IDF.txt', 'w') as filehandle:  
    for key, value in idfDict.items():
        filehandle.write('%s:%s\n' % (key, value))"""

prob_review = {}
prob_review['prob_neg_review'] = prob_neg_review
prob_review['prob_pos_review'] = prob_pos_review
prob_review['prob_neu_review'] = prob_neu_review

"""with open('wprob_neg_Output.txt', 'w') as filehandle:  
    for key, value in wprob_neg.items():
        filehandle.write('%s:%s\n' % (key, value))
with open('wprob_pos_Output.txt', 'w') as filehandle:  
    for key, value in wprob_pos.items():
        filehandle.write('%s:%s\n' % (key, value))
with open('prob_review_Output.txt', 'w') as filehandle:  
    for key, value in prob_review.items():
        filehandle.write('%s:%s\n' % (key, value))"""

with open('wprob_pos.pickle', 'wb') as handle:
    pickle.dump(wprob_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('wprob_neg.pickle', 'wb') as handle:
    pickle.dump(wprob_neg, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('wprob_neu.pickle', 'wb') as handle:
    pickle.dump(wprob_neu, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('prob_review.pickle', 'wb') as handle:
    pickle.dump(prob_review, handle, protocol=pickle.HIGHEST_PROTOCOL)
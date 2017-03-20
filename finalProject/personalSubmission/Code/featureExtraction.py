import re,sys
import os
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import random
import nltk
from collections import Counter


##################################### Features ####################################################

#returns the number of sentences in text
def count_sentences(sentences):
    number_sentences = len(sentences);
    return number_sentences;
        
#returns the number of tokens in text
def count_tokens(tokens):
    number_tokens = len(tokens);
    return number_tokens;

#returns average sentence length
def avg_sentence_length(sentences):
    total_length = 0;
    for sentence in sentences:
        total_length = total_length + count_tokens(sentence);
    return (float(total_length))/float(len(sentence));

#returns average token length
def avg_token_length(tokens):
    total_length = 0;
    for token in tokens:
        total_length = total_length + len(token);
    return (float(total_length))/(float(len(tokens)));

#returns number of semicolons
#def count_semicolons(text):
#   semi = 

#returns number of pronouns
def count_pronouns(text):
    count_pronouns = 0;
    for line in text:
        line = line.lower();
        count_pronouns = count_pronouns + len(re.findall('he|him|his|himself|she|her|hers|herself|it|its|itself',line));
    return count_pronouns;        

 #returns an array with feature values
def feature_array_function(text):
    tokens = nltk.word_tokenize(text);
    sentences = nltk.sent_tokenize(text);

    feat_array=[];
    feat_array.append(count_sentences(sentences));          #number of sentences
    feat_array.append(count_tokens(tokens));             #number of tokens
    feat_array.append(avg_sentence_length(sentences));      #average sentence length
    feat_array.append(avg_token_length(tokens));         #average token length
    feat_array.append(count_pronouns(text));

    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));
    # feat_array.append(count_tokens(tokens));

    
    return feat_array;

###################################################################
#####################bag of words##################################
###################################################################

# def bag_of_words_train(trainData):
# #bag of words features
#     count_vect = CountVectorizer();
#     X_train_counts = count_vect.fit_transform(trainingBunch.data);
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts);
# print (X_train_tfidf.shape);
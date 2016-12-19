import re,sys
import os
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
import random
import nltk
from collections import Counter


##################################### Features ####################################################

#returns the number of sentences in text
def count_sentences(text):
    sentences = nltk.sent_tokenize(text);
    number_sentences = len(sentences);
    return number_sentences;
        
#returns the number of tokens in text
def count_tokens(text):
    tokens = nltk.word_tokenize(text);
    number_tokens = len(tokens);
    return number_tokens;

#returns average sentence length
def avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text);
    total_length = 0;
    for sentence in sentences:
        total_length = total_length + count_tokens(sentence);
    return (float(total_length))/float(len(sentence));

#returns average token length
def avg_token_length(text):
    tokens = nltk.word_tokenize(text);
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
        count_pronouns += re.findall('he|him|his|himself|she|her|hers|herself|it|its|itself',line);
    return count_pronouns;        

 #returns an array with feature values
def feature_array_function(text):
    feat_array=[];
    feat_array.append(count_sentences(text));          #number of sentences
    feat_array.append(count_tokens(text));             #number of tokens
    feat_array.append(avg_sentence_length(text));      #average sentence length
    feat_array.append(avg_token_length(text));         #average token length
    feat_array.append(count_pronouns(text));
    
    return feat_array;
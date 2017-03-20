import re,sys
import os
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import random
import nltk
from collections import Counter


author_list = ['AaronPressman', 'AlanCrosby', 'AlexanderSmith', 'BenjaminKangLim', 'BernardHickey', 'BradDorfman',
               'DarrenSchuettler', 'DavidLawder', 'EdnaFernandes', 'EricAuchard', 'FumikoFujisaki', 'GrahamEarnshaw',
               'HeatherScoffield', 'JaneMacartney', 'JanLopatka', 'JimGilchrist', 'JoeOrtiz', 'JohnMastrini', 'JonathanBirt',
               'JoWinterbottom', 'KarlPenhaul', 'KeithWeir', 'KevinDrawbaugh', 'KevinMorrison', 'KirstinRidley', 'KouroshKarimkhany',
               'LydiaZajc', 'LynneO\'Donnell', 'LynnleyBrowning', 'MarcelMichelson', 'MarkBendeich', 'MartinWolk', 'MatthewBunce',
               'MichaelConnor', 'MureDickie', 'NickLouth', 'PatriciaCommins', 'PeterHumphrey', 'PierreTran', 'RobinSidel', 'RogerFillion',
               'SamuelPerry', 'SarahDavison', 'ScottHillis', 'SimonCowell', 'TanEeLyn', 'TheresePoletti', 'TimFarrand', 'ToddNissen',
               'WilliamKazer'];

#training array
X_train1 = [];
labels = [];
extra_features_train=[];
author_text = {};
pos = [];
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
    pronouns = [];
    for line in text:
        line = line.lower();
        pronouns.append(re.findall('he|him|his|himself|she|her|hers|herself|it|its|itself',line));
    return len(pronouns);        

def pos_tags(tokens):
    pos.append(" ".join( [ tag for (word, tag) in nltk.pos_tag( tokens ) ] ));
    #print(pos);    
    
#returns an array with feature values
def feature_array_function(text):
    sentences = nltk.sent_tokenize(text);
    tokens = nltk.word_tokenize(text);
    feat_array = [count_sentences(sentences),                 #number of sentences
                 count_tokens(tokens),                       #number of tokens
                 avg_sentence_length(sentences),             #number of tokens
                 avg_token_length(tokens),                   #average token length
                 count_pronouns(text)];                      #count pronouns
    
    pos_tags (tokens);
    return feat_array;

    

#############################################################################################

#Create bunch from trainingData
trainingBunch = sklearn.datasets.load_files("training/",load_content=True,encoding="cp1252",shuffle=False)
path_training = 'training/'
authors = os.listdir(path_training);
count = 0;

for auth in authors:
    if count == 0:
        count =+ 1;
        continue;

    files = os.listdir(path_training+auth+'/');
    feature_array = [];
    i = -1;
    for name in author_list:
        i += 1;
        if name == auth:
            break;
        

    for file in files:
        f = open(path_training+auth+'/'+file, 'r', encoding='cp1252');
        data = f.read();
        X_train1.append(feature_array_function(data));
        labels.append(i);
        f.close();
    

###################################################################
#####################Print Matrix##################################
###################################################################

#print(np.matrix(X_train1));

###################################################################
#####################bag of words##################################
###################################################################


#bag of words features
count_vect = CountVectorizer();
X_train_counts = count_vect.fit_transform(trainingBunch.data);
tfidf_transformer = TfidfTransformer()
X_train_2 = tfidf_transformer.fit_transform(X_train_counts);
#print (X_train_tfidf.shape);

###################################################################
#####################POS n-grams###################################
###################################################################

tag_vector_counts = count_vect.fit_transform( pos );
tfidf_transformer = TfidfTransformer()
X_train_3 = tfidf_transformer.fit_transform(tag_vector_counts);
#print (X2.shape);

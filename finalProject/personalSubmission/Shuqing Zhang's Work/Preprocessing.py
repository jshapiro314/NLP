import nltk
import numpy
import glob
import os

sentence_tokenizer = nltk.data.load('tokenizer/punk/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

#load data
#data_folder = r'C50'
#for root, subdirs, files in os.walk(data_folder):
#    list_file_path = os.path.join(root, '*.txt')

#################

#tokenization
def Tokenization(raw_text):
    tokens = nltk.word_tokenize(raw_text) #tokens inlcude punctuation
    words = word_tokenizer.tokenize(raw_text) #words exclude punctuation
    sentences = sentence_tokenizer.tokenize(raw_text)
    words_per_sentence = numpy.array([len(word_tokenizer.tokenize(s)) for s in sentences])

    return tokens, words, sentences, words_per_sentence


def PosTag(raw_text):
    tokens, words, sentences, word_per_sentence = Tokenization(raw_text)
    for i in tokens:
        tags = nltk.pos_tag(i)

from nltk import trigrams
def TrigramsModel(raw_text):
    tokens, words, sentences, word_per_sentence = Tokenization(raw_text)
    tri_grams = trigrams(tokens)
    fdist = nltk.FreqDist(tri_grams)

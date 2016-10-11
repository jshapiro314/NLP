import nltk
import string
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import PunktSentenceTokenizer

##################################################################################################
######NOTE: FOR SOME ANSWERS PRINT STATEMENTS NEED TO BE COMMENTED OUT. LOOK FOR ######## COMMENTS
##################################################################################################

##FROM http://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

#Read the raw text into a list
tokenList = nltk.word_tokenize(gutenberg.raw("austen-emma.txt"))

#Print length of text (list) before data clensing
print("Number of words in text: " , len(tokenList))
#print(tokenList)

#Import punctuations and stop words, remove them from text file.
#I ASSUME WE ARE REMOVING PUNCTUATION TOKENS AND STOP WORDS, NOT TOKENS THAT INCLUDE PUNCTUATION (like ca n't)
cleanedList = []
stopWords = set(stopwords.words("English"))
for m in tokenList:
	if m not in string.punctuation:
		n = m.lower()
		if n not in stopWords:
			cleanedList.append(m)

#Print length of cleaned text (list)
print("Number of words in cleaned text:" , len(cleanedList))
#print(cleanedList)

#Save cleaned list as text file
cleanedFile = open("cleaned.txt", "w")
for word in cleanedList:
	cleanedFile.write(word)
	cleanedFile.write(" ")

#Stem and lemmatize, using POS tag

#Stemming
stemmer = PorterStemmer()
stemmedList = []
for words in cleanedList:
	stemmedList.append(stemmer.stem(words))
#######COMMENT OUT PRINT TO SEE STEMMED LIST#######
#print(stemmedList)

#lemmatization (ONLY ADDING TO LEMMATIZED LIST IF WORD TAGGED WITH WORDNET POS)
lemmatizedList = []
lemmatizer = WordNetLemmatizer()
pst = PunktSentenceTokenizer()
pstTokens = pst.tokenize(gutenberg.raw("austen-emma.txt"))
for item in pstTokens:
	word = nltk.word_tokenize(item)
	pos = nltk.pos_tag(word)
	for group in pos:
		realPos = get_wordnet_pos(group[1])
		if realPos != "":
			lemmatizedList.append(lemmatizer.lemmatize(group[0], realPos))
#######COMMENT OUT PRINT TO SEE LEMMATIZED LIST#######
#print(lemmatizedList)


#Print number of sentences using sentence tokenizer
sentenceList = nltk.sent_tokenize(gutenberg.raw("austen-emma.txt"))
print("Number of sentences in text: " , len(sentenceList))

#Randomly select 5 sentences from text and print/draw the NER tags
#TESTING FIRST 5, but could be changed to random easily
i=0
while i < 5:
	word = nltk.word_tokenize(pstTokens[i])
	pos = nltk.pos_tag(word)
	ne_word = nltk.ne_chunk(pos)
	#######COMMENT OUT PRINT TO SEE NER CHUNKS#######
	#print(ne_word)
	i += 1

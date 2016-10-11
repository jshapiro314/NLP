import nltk
from nltk.corpus import wordnet
#Read clean.txt and find top 20 unigrans and bigrams using frequency of unigrams and append to a set
#Create synonyms and antonyms for unigram sets using Wordnet synset.

#Read in cleaned.txt file
cleanedFile = open("cleaned.txt", "r")
contents = cleanedFile.read()
tokenList = nltk.word_tokenize(contents)
tokenList = [x.lower() for x in tokenList]

#Calculate top 20 unigrams
unigramList = nltk.FreqDist(tokenList).most_common(20)
#print(unigramList)

#Calculate top 20 bigrams
bigramList = nltk.FreqDist(nltk.bigrams(tokenList)).most_common(20)
#print(bigramList)

#Calculate synonyms & antonyms for unigrams
synonyms = {}
antonyms = {}

synonymList = []
antonymList = []

for word in unigramList:
	for items in wordnet.synsets(word[0]):
		for l in items.lemmas():
			synonymList.append(l.name())
			if l.antonyms():
				antonymList.append(l.antonyms()[0].name())

	synonyms[word] = synonymList
	antonyms[word] = antonymList

	synonymList = []
	antonymList = []

print(synonyms)
print(antonyms)
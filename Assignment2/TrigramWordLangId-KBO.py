import re
from calculatePerplexity import calcPerplexityTrigram
from trigramFuncs import *

#Removing some punctuation (not including - or ') and splitting words on whitespace. Not changing uppercase letters to lowercase
#I am not using the start or end of sentences as tokens
#I am not normalizing probabilities, so if I don't find the trigram, I simply use the probability of the bigram, NOT the probability x lambda.
#If a word is in the test data that isn't present in the training data, the probability of that word would become 0, making the sentence 0. To fix this problem without
#implementing a form of smoothing, I've assigned a probability of 0.0000000001 for the words that would have a probability of 0. The smaller this number gets, the more
#accurate the results are. This number provided yields very promising classification results, however it skews perplexity extremely high.


#open files and tokenize them
engTokens = tokenizeInput("HW2english.txt")
frTokens = tokenizeInput("HW2french.txt")
gerTokens = tokenizeInput("HW2german.txt")

#calculate counts of unigrams, bigrams, and trigrams
engModel = createModelCounts(engTokens)
frModel = createModelCounts(frTokens)
gerModel = createModelCounts(gerTokens)

#calculate probabilies of unigrams, bigrams, and trigrams
creatModelProbabilities(engModel)
creatModelProbabilities(frModel)
creatModelProbabilities(gerModel)

#calculate probabilities of each sentence in test data and select max for guessed language (choose the first language if tie)
testFile = open('LangID.test.txt', 'r')
outputFile = open('TrigramWordLangId-KBO.out', 'w')
outputFile.write("ID LANG\n")
engProb = 1
frProb = 1
gerProb = 1
maxProb = 0
count = 1
outputVal = ""

testStrings = testFile.readlines()
#used for calculating perplexity
testData = []
for line in testStrings:
	#print(line)
	#remove number from front of line
	line = re.sub('^[\\d]+\\.\\s', '', line)
	#remove same punctuation as in training data
	line = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»_|]', '', line)
	#tokenize
	lineTokens = re.split('\\s', line)
	#print(line)
	outputVal  = str(count) + '. '
	count += 1
	for i in range(0,len(lineTokens)-2):
		testData.append(lineTokens[i])
		engProb *= calcProbability (i, lineTokens, engModel)
		frProb *= calcProbability (i, lineTokens ,frModel)
		gerProb *= calcProbability (i, lineTokens, gerModel)
	
	# print(engProb)
	# print(frProb)
	# print(gerProb)
	testData.append(lineTokens[len(lineTokens)-1])
	maxProb = max(engProb, frProb, gerProb)

	if engProb == maxProb:
		outputVal = outputVal + 'EN\n'
	elif frProb == maxProb:
		outputVal = outputVal + 'FR\n'
	else:
		outputVal = outputVal + 'GR\n'
	
	outputFile.write(outputVal)
	engProb = 1
	frProb = 1
	gerProb = 1
	maxProb = 0
	outputVal = ""


#output perplexity
print("Eng perplexity:")
print(calcPerplexityTrigram(engModel, testData))
print("Fr perplexity:")
print(calcPerplexityTrigram(frModel, testData))
print("Ger perplexity:")
print(calcPerplexityTrigram(gerModel, testData))



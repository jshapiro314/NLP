import re
from calculatePerplexity import calcPerplexity

#Removing some punctuation (not including - or ') and splitting words on whitespace. Not changing uppercase letters to lowercase
#I am not using the start or end of sentences as tokens
#I am not normalizing probabilities, so if I don't find the trigram, I simply use the probability of the bigram, NOT the probability x lambda.
#I am not performing any smoothing. If the unigram does not exist in the training data, the probability will be 0.


#Read in files and tokenize
engTokens = tokenizeInput("HW2english.txt")

#Create language models

engModel = LanguageModel()




#Function that calculates the unigram, bigram, and trigram probabilities based on a LanguageModel object that only includes counts.
def creatModelProbabilities ( model ):
	#first convert trigram to probabilities
	for x in model.trigram.keys():
		a,b,c = x
		model.trigram[x] = model.trigram[x] / model.bigram[(a,b)]

	#then convert bigram to probabilities
	for x in model.bigram.keys():
		a,b = x
		model.bigram[x] = model.bigram[x] / model.unigram[a]

	#finally convert unigram to probabilities
	for x in model.unigram.keys():
		model.unigram[x] = model.unigram[x] / model.tokenNumber

#Function that returns the unigram, bigram, and trigram frequency model for a list of tokens
def createModelCounts ( tokens ):
	tokenNumber = len(tokens)
	monogram = {}
	bigram = {}
	trigram = {}
	for i in range(0,len(tokens)-2):
		if tokens[i] not in monogram.keys():
			monogram[tokens[i]] = 1
		else:
			monogram[tokens[i]] += 1

		if (tokens[i],tokens[i+1]) not in bigram.keys():
			bigram[(tokens[i],tokens[i+1])] = 1
		else:
			bigram[(tokens[i],tokens[i+1])] += 1

		if (tokens[i],tokens[i+1],tokens[i+2]) not in monogram.keys():
			trigram[(tokens[i],tokens[i+1],tokens[i+2])] = 1
		else:
			trigram[(tokens[i],tokens[i+1],tokens[i+2])] += 1

	if tokens[len(tokens)-2] not in monogram.keys():
		monogram[tokens[len(tokens)-2]] = 1
	else:
		monogram[tokens[len(tokens)-2]] += 1
	if tokens[len(tokens)-1] not in monogram.keys():
		monogram[tokens[len(tokens)-1]] = 1
	else:
		monogram[tokens[len(tokens)-1]] += 1
	if (tokens[len(tokens)-2],tokens[len(tokens)-1]) not in bigram.keys():
		bigram[(tokens[len(tokens)-2],tokens[len(tokens)-1])] = 1
	else:
		bigram[(tokens[len(tokens)-2],tokens[len(tokens)-1])] += 1

	model = LanguageModel(tokenNumber, monogram, bigram, trigram)
	return model


#Function that returns the tokens from the training data after removing some punctuation and splitting on whitespace
#input:fileName
def tokenizeInput ( fileName ):
	text = open(fileName, 'r')
	textString = text.read()
	textString = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»_|]', '', textString)
	tokens = re.split('\\s', textString)
	return tokens


#Function that returns the given probability of the trigram/bigram/monogram given the index of the beginning of the trigram
#input:index, monogram, bigram, and trigram models, and tokens
def calcProbability ( index , tokens , model ):
	probability = 0

	if (tokens[index], tokens[index+1], tokens[index+2]) in model.trigram.keys():
		probability = model.trigram[(tokens[index], tokens[index+1], tokens[index+2])]
	elif (tokens[index+1], tokens[index+2]) in model.bigram.keys():
		probability = model.bigram[(tokens[index+1], tokens[index+2])]
	elif tokens[index+2] in model.monogram.keys():
		probability = model.monogram[tokens[index+2]]
	else:
		probability = 0


	return probability

#Class to hold language models. Includes unigram, bigram, and trigram
class LanguageModel:
	def __init__(self, tokenNumber, unigram, bigram, trigram):
		self.tokenNumber = tokenNumber
		self.unigram = unigram
		self.bigram = bigram
		self.trigram = trigram
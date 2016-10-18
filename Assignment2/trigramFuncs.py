import re

#Functions that are called in TrigramWordLandId-KBO.py


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
	unigram = {}
	bigram = {}
	trigram = {}
	for i in range(0,len(tokens)-2):
		if tokens[i] not in unigram.keys():
			unigram[tokens[i]] = 1
		else:
			unigram[tokens[i]] += 1

		if (tokens[i],tokens[i+1]) not in bigram.keys():
			bigram[(tokens[i],tokens[i+1])] = 1
		else:
			bigram[(tokens[i],tokens[i+1])] += 1

		if (tokens[i],tokens[i+1],tokens[i+2]) not in unigram.keys():
			trigram[(tokens[i],tokens[i+1],tokens[i+2])] = 1
		else:
			trigram[(tokens[i],tokens[i+1],tokens[i+2])] += 1

	if tokens[len(tokens)-2] not in unigram.keys():
		unigram[tokens[len(tokens)-2]] = 1
	else:
		unigram[tokens[len(tokens)-2]] += 1
	if tokens[len(tokens)-1] not in unigram.keys():
		unigram[tokens[len(tokens)-1]] = 1
	else:
		unigram[tokens[len(tokens)-1]] += 1
	if (tokens[len(tokens)-2],tokens[len(tokens)-1]) not in bigram.keys():
		bigram[(tokens[len(tokens)-2],tokens[len(tokens)-1])] = 1
	else:
		bigram[(tokens[len(tokens)-2],tokens[len(tokens)-1])] += 1

	model = LanguageModel(tokenNumber, unigram, bigram, trigram)
	return model


#Function that returns the tokens from the training data after removing some punctuation and splitting on whitespace
#input:fileName
def tokenizeInput ( fileName ):
	text = open(fileName, 'r')
	textString = text.read()
	textString = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»_|]', '', textString)
	tokens = re.split('\\s', textString)
	return tokens


#Function that returns the given probability of the trigram/bigram/unigram given the index of the beginning of the trigram
#input:index, unigram, bigram, and trigram models, and tokens
def calcProbability ( index , tokens , model ):
	probability = 0

	if (tokens[index], tokens[index+1], tokens[index+2]) in model.trigram.keys():
		probability = model.trigram[(tokens[index], tokens[index+1], tokens[index+2])]
	elif (tokens[index+1], tokens[index+2]) in model.bigram.keys():
		probability = model.bigram[(tokens[index+1], tokens[index+2])]
	elif tokens[index+2] in model.unigram.keys():
		probability = model.unigram[tokens[index+2]]
	else:
		probability = 0.0000000001


	return probability

#Class to hold language models. Includes unigram, bigram, and trigram
class LanguageModel:
	def __init__(self, tokenNumber, unigram, bigram, trigram):
		self.tokenNumber = tokenNumber
		self.unigram = unigram
		self.bigram = bigram
		self.trigram = trigram
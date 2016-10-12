import re
from math import pow, log2

#This function takes in a language model (model) which consists of a dictionary of dictionaries that show the probabilities of bigrams.
#This function also takes in a test data set (testData) which is a list of tokens that are located in the language model. This list has already been cleaned so that this function only has to worry about calculating perplexity
#The function returns the perplexity value, or the word infinity if the probability of the testData = 0 (because we can't perform 1/0 on line 15).
def calcPerplexity ( model , testData ):
	perplexity = 1
	for i in range(0,len(testData)-1):
		if testData[i] in model.keys() and testData[i+1] in model.keys() and model[testData[i]][testData[i+1]] > 0:
			#print(model[testData[i]][testData[i+1]])
			perplexity = perplexity * pow((1 / model[testData[i]][testData[i+1]]), 1 / len(testData))
			#print(perplexity)
		else:
			return "infinity"


	return perplexity
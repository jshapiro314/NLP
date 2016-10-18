import re
from calculatePerplexity import calcPerplexityBigram

#Removing some punctuation (not including - or ') and splitting words on whitespace. Not changing uppercase letters to lowercase
#I am not using the start or end of sentences as tokens

#Used the following link for good turing smoothing guidance: http://rstudio-pubs-static.s3.amazonaws.com/165358_78fd356d6e124331bd66981c51f7ad7c.html
#I am only using good turing smoothing to modify my N0 though, because I don't want to run into the issue of getting counts of 0 when N(c+1) = 0.
#I am choosing not to normalize my probabilities.


#Open files, read in data, remove punctuation not including - or '
engText = open('HW2english.txt', 'r')
engTextString = engText.read()
engTextString = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»_|]', '', engTextString)
engDict = {}

frText = open('HW2french.txt', 'r')
frTextString = frText.read()
frTextString = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»_|]', '', frTextString)
frDict = {}

gerText = open('HW2german.txt', 'r')
gerTextString = gerText.read()
gerTextString = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»_|]', '', gerTextString)
gerDict = {}

#tokenize on spaces
engTokens = re.split('\\s', engTextString)
frTokens = re.split('\\s', frTextString)
gerTokens = re.split('\\s', gerTextString)


#get list of unique words from test set as well, and add to each language's tokens. This ensures that we never come across a word we haven't seen before.
#Initialize the count of these words to 0, since they may not be seen in the training data.
testFile = open('LangID.test.txt', 'r')
testString = testFile.readlines()
for line in testString:
	line = re.sub('^[\\d]+\\.\\s', '', line)
	line = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»_|]', '', line)
	lineTokens = re.split('\\s', line)
	for c in lineTokens:
		if c not in engDict.keys():
			engDict[c] = 0
		if c not in frDict.keys():
			frDict[c] = 0
		if c not in gerDict.keys():
			gerDict[c] = 0


#get list of unique words and their frequency from training sets
for c in engTokens:
	if c not in engDict.keys():
		engDict[c] = 1
	else:
		engDict[c] += 1

for c in frTokens:
	if c not in frDict.keys():
		frDict[c] = 1
	else:
		frDict[c] += 1

for c in gerTokens:
	if c not in gerDict.keys():
		gerDict[c] = 1
	else:
		gerDict[c] += 1
#print(gerDict.keys())

#By here we have a dictionary for each language that includes tokens and frequencies.

#build dictionary of dictionary of unique words x unique words & set initial value to 0
engModel = {}
for x in engDict.keys():
	engModel[x] = {}
	for y in engDict.keys():
		engModel[x][y] = 0

frModel = {}
for x in frDict.keys():
	frModel[x] = {}
	for y in frDict.keys():
		frModel[x][y] = 0

gerModel = {}
for x in gerDict.keys():
	gerModel[x] = {}
	for y in gerDict.keys():
		gerModel[x][y] = 0

#print(engModel)

#populate dictionaries by iterating over each word
for i in range(0,len(engTokens)-1):
	engModel[engTokens[i]][engTokens[i+1]] += 1

for i in range(0,len(frTokens)-1):
	frModel[frTokens[i]][frTokens[i+1]] += 1

for i in range(0,len(gerTokens)-1):
	gerModel[gerTokens[i]][gerTokens[i+1]] += 1

#print(engModel['t']['h'])
#print(engDict['t'])

#By here we have a table of bigrams and the frequencies of the bigrams for each language
#It's time to calculate the new counts for our 0 occurence bigrams.

engN0 = 0
engN1 = 0
engN = 0

frN0 = 0
frN1 = 0
frN = 0

gerN0 = 0
gerN1 = 0
gerN = 0

for x in engDict.keys():
	for y in engDict.keys():
		engN += engModel[x][y]
		if engModel[x][y] == 0:
			engN0 += 1
		elif engModel[x][y] == 1:
			engN1 += 1

for x in frDict.keys():
	for y in frDict.keys():
		frN += frModel[x][y]
		if frModel[x][y] == 0:
			frN0 += 1
		elif frModel[x][y] == 1:
			frN1 += 1

for x in gerDict.keys():
	for y in gerDict.keys():
		gerN += gerModel[x][y]
		if gerModel[x][y] == 0:
			gerN0 += 1
		elif gerModel[x][y] == 1:
			gerN1 += 1

eng0Prob = engN1 / (engN0 * engN)
fr0Prob = frN1 / (frN0 * frN)
ger0Prob = gerN1 / (gerN0 * gerN)


#divide each row by frequency of each word or if value == 0, sub in with probability generated above
for x in engDict.keys():
	for y in engModel[x].keys():
		if engModel[x][y] == 0:
			engModel[x][y] = eng0Prob
		else:
			engModel[x][y] = engModel[x][y] / engDict[x]


for x in frDict.keys():
	for y in frModel[x].keys():
		if frModel[x][y] == 0:
			frModel[x][y] = fr0Prob
		else:
			frModel[x][y] = frModel[x][y] / frDict[x]

for x in gerDict.keys():
	for y in gerModel[x].keys():
		if gerModel[x][y] == 0:
			gerModel[x][y] = ger0Prob
		else:
			gerModel[x][y] = gerModel[x][y] / gerDict[x]

#print(engModel['t']['h'])
#I've brought lots of wine

#print(engModel['''I've''']["brought"])
#print(engModel["brought"]["lots"])
#print(engModel["lots"]["of"])
#print(engModel["of"]["wine"])

#calculate probabilities of each sentence in test data and select max for guessed language (choose the first language if tie)
testFile = open('LangID.test.txt', 'r')
outputFile = open('BigramWordLangId-GT.out', 'w')
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
	for i in range(0,len(lineTokens)-1):
		testData.append(lineTokens[i])
		engProb *= engModel[lineTokens[i]][lineTokens[i+1]]
		frProb *= frModel[lineTokens[i]][lineTokens[i+1]]
		gerProb *= gerModel[lineTokens[i]][lineTokens[i+1]]
	#print(engProb)
	#print(frProb)
	#print(gerProb)
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
print(calcPerplexityBigram(engModel, testData))
print("Fr perplexity:")
print(calcPerplexityBigram(frModel, testData))
print("Ger perplexity:")
print(calcPerplexityBigram(gerModel, testData))

import re

#Removing some punctuation (not including - or ') and splitting words on whitespace.
#While add 1 smoothing accounts for the bigrams made from the training data that weren't located in the training data,
#it does not easily handle new words in the test set that aren't located in the training data. Instead of looking at
#the test set, I am simply adding an extra word to my bigrams called <UNKOWN>. If I find a word in the test set that
#isn't in the training set, I substitute out the word for <UNKOWN> and calculate the bigram probability based on that.


#Open files, read in data, remove punctuation not including - or '
engText = open('HW2english.txt', 'r')
engTextString = engText.read()
engTextString = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»]', '', engTextString)
engDict = {}

frText = open('HW2french.txt', 'r')
frTextString = frText.read()
frTextString = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»]', '', frTextString)
frDict = {}

gerText = open('HW2german.txt', 'r')
gerTextString = gerText.read()
gerTextString = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»]', '', gerTextString)
gerDict = {}

#tokenize on spaces
engTokens = re.split('\\s', engTextString)
frTokens = re.split('\\s', frTextString)
gerTokens = re.split('\\s', gerTextString)

#get list of unique words and their frequency & add <UNKNOWN>
for c in engTokens:
	if c not in engDict.keys():
		engDict[c] = 1
	else:
		engDict[c] += 1
engDict["<UNKNOWN>"] = 0

for c in frTokens:
	if c not in frDict.keys():
		frDict[c] = 1
	else:
		frDict[c] += 1
frDict["<UNKNOWN>"] = 0

for c in gerTokens:
	if c not in gerDict.keys():
		gerDict[c] = 1
	else:
		gerDict[c] += 1
gerDict["<UNKNOWN>"] = 0
#print(gerDict.keys())

#Add the number of tokens to the frequency of each token (part of add 1 smoothing)
for key in engDict.keys():
	engDict[key] += len(engDict.keys())

for key in frDict.keys():
	frDict[key] += len(frDict.keys())

for key in gerDict.keys():
	gerDict[key] += len(gerDict.keys())


#build dictionary of dictionary of unique words x unique words & set initial value to 1 (for add 1 smoothing)
engModel = {}
for x in engDict.keys():
	engModel[x] = {}
	for y in engDict.keys():
		engModel[x][y] = 1

frModel = {}
for x in frDict.keys():
	frModel[x] = {}
	for y in frDict.keys():
		frModel[x][y] = 1

gerModel = {}
for x in gerDict.keys():
	gerModel[x] = {}
	for y in gerDict.keys():
		gerModel[x][y] = 1

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

#divide each row by frequency of each word (includes number of tokens as well)
for x in engDict.keys():
	for y in engModel[x].keys():
		engModel[x][y] = engModel[x][y] / engDict[x]

for x in frDict.keys():
	for y in frModel[x].keys():
		frModel[x][y] = frModel[x][y] / frDict[x]

for x in gerDict.keys():
	for y in gerModel[x].keys():
		gerModel[x][y] = gerModel[x][y] / gerDict[x]

#print(engModel['t']['h'])
#I've brought lots of wine

#print(engModel['''I've''']["brought"])
#print(engModel["brought"]["lots"])
#print(engModel["lots"]["of"])
#print(engModel["of"]["wine"])

#calculate probabilities of each sentence in test data and select max for guessed language (choose the first language if tie)
testFile = open('LangID.test.txt', 'r')
outputFile = open('BigramWordLangId-AO.out', 'w')
outputFile.write("ID LANG\n")
engProb = 1
frProb = 1
gerProb = 1
maxProb = 0
count = 1
outputVal = ""

testStrings = testFile.readlines()
for line in testStrings:
	#print(line)
	#remove number from front of line
	line = re.sub('^[\\d]+\\.\\s', '', line)
	#remove same punctuation as in training data
	line = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»]', '', line)
	#tokenize
	lineTokens = re.split('\\s', line)
	#print(line)
	outputVal  = str(count) + '. '
	count += 1
	for i in range(0,len(lineTokens)-1):
		if lineTokens[i] in engModel.keys() and lineTokens[i+1] in engModel.keys():
			engProb *= engModel[lineTokens[i]][lineTokens[i+1]]
		elif lineTokens[i] in engModel.keys() and lineTokens[i+1] not in engModel.keys():
			engProb *= engModel[lineTokens[i]]["<UNKNOWN>"]
		elif lineTokens[i] not in engModel.keys() and lineTokens[i+1] in engModel.keys():
			engProb *= engModel["<UNKNOWN>"][lineTokens[i+1]]
		else:
			engProb *= engModel["<UNKNOWN>"]["<UNKNOWN>"]

		if lineTokens[i] in frModel.keys() and lineTokens[i+1] in frModel.keys():
			frProb *= frModel[lineTokens[i]][lineTokens[i+1]]
		elif lineTokens[i] in frModel.keys() and lineTokens[i+1] not in frModel.keys():
			frProb *= frModel[lineTokens[i]]["<UNKNOWN>"]
		elif lineTokens[i] not in frModel.keys() and lineTokens[i+1] in frModel.keys():
			frProb *= frModel["<UNKNOWN>"][lineTokens[i+1]]
		else:
			frProb *= frModel["<UNKNOWN>"]["<UNKNOWN>"]

		if lineTokens[i] in gerModel.keys() and lineTokens[i+1] in gerModel.keys():
			gerProb *= gerModel[lineTokens[i]][lineTokens[i+1]]
		elif lineTokens[i] in gerModel.keys() and lineTokens[i+1] not in gerModel.keys():
			gerProb *= gerModel[lineTokens[i]]["<UNKNOWN>"]
		elif lineTokens[i] not in gerModel.keys() and lineTokens[i+1] in gerModel.keys():
			gerProb *= gerModel["<UNKNOWN>"][lineTokens[i+1]]
		else:
			gerProb *= gerModel["<UNKNOWN>"]["<UNKNOWN>"]

	#print(engProb)
	#print(frProb)
	#print(gerProb)
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





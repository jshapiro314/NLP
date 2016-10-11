import re

#Removing punctuation other than - and '. Also removing \n from end of each sentence in test set.


#Open files and read in data
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

#get list of unique characters and their frequency
for c in engTextString:
	if c not in engDict.keys():
		engDict[c] = 1
	else:
		engDict[c] += 1

for c in frTextString:
	if c not in frDict.keys():
		frDict[c] = 1
	else:
		frDict[c] += 1

for c in gerTextString:
	if c not in gerDict.keys():
		gerDict[c] = 1
	else:
		gerDict[c] += 1
#print(gerDict.keys())

#build dictionary of dictionary of unique characters x unique characters

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

#populate dictionaries by iterating over each character
for i in range(0,len(engTextString)-1):
	engModel[engTextString[i]][engTextString[i+1]] += 1

for i in range(0,len(frTextString)-1):
	frModel[frTextString[i]][frTextString[i+1]] += 1

for i in range(0,len(gerTextString)-1):
	gerModel[gerTextString[i]][gerTextString[i+1]] += 1

#print(engModel['t']['h'])
#print(engDict['t'])

#divide each row by frequency of each character to get probabilities
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


#calculate probabilities of each sentence in test data and select max for guessed language (choose the first language if tie)
testFile = open('LangID.test.txt', 'r')
outputFile = open('BigramLetterLangId.out', 'w')
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
	line = re.sub('^[\\d]+\\.\\s', '', line)
	line = re.sub('[\"\\.!(),?;:\\[\\]{}@#$%^&*+=\\\\\\/<>«»_|\n]', '', line)
	#print(line)
	outputVal  = str(count) + '. '
	count += 1
	for i in range(0,len(line)-1):
		if line[i] in engModel.keys() and line[i+1] in engModel.keys():
			engProb *= engModel[line[i]][line[i+1]]
		else:
			engProb = 0

		if line[i] in frModel.keys() and line[i+1] in frModel.keys():
			frProb *= frModel[line[i]][line[i+1]]
		else:
			frProb = 0
		if line[i] in gerModel.keys() and line[i+1] in gerModel.keys():
			gerProb *= gerModel[line[i]][line[i+1]]
		else:
			gerProb = 0

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





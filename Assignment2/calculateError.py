import re

def calculatePercentage ( textFile ):

	#compare tokens for each line in each file to figure out if we match the gold standard
	goldFile = open('LangID.gold.txt', 'r')
	inputFile = open(textFile, 'r')
	print(textFile)

	goldText = goldFile.readlines()
	inputText = inputFile.readlines()

	model = {'EN':{'EN':0, 'FR':0, 'GR':0}, 'FR':{'EN':0, 'FR':0, 'GR':0}, 'GR':{'EN':0, 'FR':0, 'GR':0}}

	count = 0
	for i in range(1,len(goldText)):
		goldTokens = re.split('\\s', goldText[i])
		inputTokens = re.split('\\s', inputText[i])

		if goldTokens == inputTokens:
			count += 1
		else:
			model[inputTokens[1]][goldTokens[1]] += 1

	print(count/150)
	print(model)

	return

calculatePercentage("BigramLetterLangId.out")
calculatePercentage("BigramWordLangId-AO.out")
calculatePercentage("BigramWordLangId-GT.out")
calculatePercentage("TrigramWordLangId-KBO.out")



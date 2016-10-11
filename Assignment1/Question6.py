import re
#parse text and place into list
pattern = "(([0-9]+[0-9\\.,]*[0-9]+)|(['’][a-z]*)|(([A-Z]\\.)+[A-Z])|([A-Z][a-z]+\\.)|(\\.{3})|([A-Za-z]+)|(\\S))"
text = """"Predictions suggesting that large changes in weight will accumulate indefinitely in response to small sustained lifestyle modifications rely on the half-century-old 3,500 calorie rule, which equates a weight alteration of 2.2 lb to a 3,500 calories cumulative deficit or increment," write the study authors Dr. Jampolis, Dr. Chaudry, and Prof. Harlen, from N.P.C Clinic in OH. The 3,500-calorie rule "predicts that a person who increases daily energy expenditure by 100 calories by walking 1 mile per day" will lose 50 pounds over five years, the authors say. But the true weight loss is only about 10 pounds if calorie intake doesn't increase, "because changes in mass ... alter the energy requirements of the body’s make-up." "This is a myth, strictly speaking, but the smaller amount of weight loss achieved with small changes is clinically significant and should not be discounted," says Dr. Melina Jampolis, CNN diet and fitness expert."""

groups = (re.findall(pattern, text))
myTokens = []
#fetch first item of each group
i=0
while(i < len(groups)):
	myTokens.append(groups[i][0])
	i+=1

#sort list
#print(myTokens)
myTokens.sort()
#print(myTokens)

#iterate over list, calculating total and printing each item
itemCount = 0
tokenCount = len(myTokens)
typeCount = 0
i=0
while i < tokenCount:
	itemCount = myTokens.count(myTokens[i])
	print(myTokens[i], " ", itemCount)
	i+=itemCount
	typeCount+=1
print("Tokens: ", tokenCount)
print("Types: ", typeCount)

import re


#Numbers: [0-9]+[0-9\.,]*[0-9]+ (includes commas and decimals in numbers as one token ie, 3,500)
#Contractions: '’[a-z]* (tokenizes can't to can 't )
#Abreviations and Titles: (([A-Z]\.)+[A-Z])|([A-Z][a-z]+\.)|(\.{3}) (handles Dr., Prof. as well as A.B.C and ...)
#Words: [A-Za-z]+
#Punctuation & anything else: \S

#([0-9]+[0-9\.,]*[0-9]+)|(['’][a-z]*)|(([A-Z]\.)+[A-Z])|([A-Z][a-z]+\.)|(\.{3})|([A-Za-z]+)|(\S)

pattern = "([0-9]+[0-9\\.,]*[0-9]+)|(['’][a-z]*)|(([A-Z]\\.)+[A-Z])|([A-Z][a-z]+\\.)|(\\.{3})|([A-Za-z]+)|(\\S)"
text = """"Predictions suggesting that large changes in weight will accumulate indefinitely in response to small sustained lifestyle modifications rely on the half-century-old 3,500 calorie rule, which equates a weight alteration of 2.2 lb to a 3,500 calories cumulative deficit or increment," write the study authors Dr. Jampolis, Dr. Chaudry, and Prof. Harlen, from N.P.C Clinic in OH. The 3,500-calorie rule "predicts that a person who increases daily energy expenditure by 100 calories by walking 1 mile per day" will lose 50 pounds over five years, the authors say. But the true weight loss is only about 10 pounds if calorie intake doesn't increase, "because changes in mass ... alter the energy requirements of the body’s make-up." "This is a myth, strictly speaking, but the smaller amount of weight loss achieved with small changes is clinically significant and should not be discounted," says Dr. Melina Jampolis, CNN diet and fitness expert."""

for m in re.finditer(pattern, text):
	print(m)



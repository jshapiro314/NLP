Joshua Shapiro
jshapiro314@gwu.edu
18 October 2016

Assignment 2

Organiziation:

Question 1 -- Soundex FST:
	File: Question1.pdf
	Contains the FST for the Soundex Algorithm. A description of the solution is given at the top of the file

Question 2 -- Bigram Letter Model:
	File: letterLangId.py
	I removed punctuation other than - and '. Also removed \n from end of each sentence in test set. 
	Not changing uppercase letters to lowercase.
	I am not using the start or end of sentences as tokens.
	Output: BigramLetterLangId.out
	Also outputs the perplexity scores of each language. In this case, it makes sense for perplexity to = infinity, since some letters in the test set are not in the training sets

Question 3 -- Bigram Word Model Add-One Smoothing:
	File: BigramWordLangId-AO.py
	I removed some punctuation (not including - or ') and split words on whitespace.
	Not changing uppercase letters to lowercase.
	I am not using the start or end of sentences as tokens.
	Output: BigramWordLangId-AO.out
	Also outputs the perplexity scores of each language.
	Note: Takes some time to run.

Question 4 -- Bigram Word Model Good-Turing Smoothing:
	File: BigramWordLangId-GT.py
	I removed some punctuation (not including - or ') and split words on whitespace.
	Not changing uppercase letters to lowercase.
	I am not using the start or end of sentences as tokens.
	I used the following link for good turing smoothing guidance: http://rstudio-pubs-static.s3.amazonaws.com/165358_78fd356d6e124331bd66981c51f7ad7c.html
	I am only using good turing smoothing to modify my N0 though, because I don't want to run into the issue of getting counts of 0 when N(c+1) = 0.
	I am choosing not to normalize my probabilities.
	Output: BigramWordLangId-GT.out
	Note: Takes some time to run.

Question 5 -- Trigram Word Model Katz Back-off:
	File: TrigramWordLangId-KBO.py
	I removed some punctuation (not including - or ') and split words on whitespace.
	Not changing uppercase letters to lowercase.
	I am not using the start or end of sentences as tokens.
	I am not normalizing probabilities, so if I don't find the trigram, I simply use the probability of the bigram, NOT the probability x lambda.
	If a word is in the test data that isn't present in the training data, the probability of that word would become 0, making the sentence 0. To fix this problem without implementing a form of smoothing, I've assigned a probability of 0.0000000001 for the words that would have a probability of 0. The smaller this number gets, the more accurate the results are. This number provided yields very promising classification results, however it skews perplexity extremely high.
	Output: TrigramWordLangId-KBO.out

	File: trigramFuncs.py
	Holds all functions and classes that are used by TrigramWordLangId-KBO.py. 

Question 6 -- Quantitative Error Analysis:
	File: Question6.pdf
	Includes error analysis of questions 2-5.

Other Files: 

	*.out Files
		All files ending in .out are the outputs of the code in questions 2-5. Their format is sentence number followed by predicted language. This is so that they match the format of the LangID.gold.txt file for easy comparison.

	CalculatePerplexity.py
		Includes functions to calculate perplexity on bigram and trigram models.
		These functions are called by the python files mentioned in questions 2-5.
		Used when creating the pdf for question 6.

	CalculateError.py
		Calculates the percentage error of each model's output compared to the gold standard and produces confusion matrix tables.
		This file needs to be called AFTER running the code for questions 2-5, as it requires the *.out files to exist.
		Used when creating the pdf for question 6.



System Operation: I was running all unix commands on a Mac running macOS 10.12 Sierra. All python code was compiled and run using Python 3.5.2.
To run python code, use python3 <nameOfFile>.py
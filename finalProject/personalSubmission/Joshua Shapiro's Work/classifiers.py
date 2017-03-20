import numpy
import nltk
import scipy
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

#Methods that handle training and testing classifiers
#Methods that evaluate accuracy over data



def svmTrainAndClassify(trainData, trainLabels, testData, testLabels, cVal, kernelVal, degreeVal):
	classifier = sklearn.svm.SVC(C = cVal, kernel = kernelVal, degree = degreeVal, decision_function_shape='ovr')
	classifier.fit(trainData,trainLabels)
	guessedLabels = classifier.predict(testData)
	return numpy.mean(guessedLabels == testLabels)


def multNBTrainAndClassify(trainData, trainLabels, testData, testLabels):
	classifier = MultinomialNB()
	classifier.fit(trainData,trainLabels)
	guessedLabels = classifier.predict(testData)
	return numpy.mean(guessedLabels == testLabels)


def gaussNBTrainAndClassify(trainData, trainLabels, testData, testLabels):
	classifier = GaussianNB()
	classifier.fit(trainData,trainLabels)
	guessedLabels = classifier.predict(testData)
	return numpy.mean(guessedLabels == testLabels)
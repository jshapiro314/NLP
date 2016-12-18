import numpy
import nltk
import scipy
import sklearn

#Methods that handle training and testing classifiers
#Methods that evaluate accuracy over data



def svmTrainAndClassify(trainData, trainLabels, testData, testLabels, cVal, kernelVal, degreeVal):
	classifier = sklearn.svm.SVC(C = cVal, kernel = kernelVal, degree = degreeVal)
	classifier.fit(trainData,trainLabels)
	guessedLabels = classifier.predict(testData)
	return numpy.mean(guessedLabels == testLabels)


def multNBTrainAndClassify(traindata, trainLabels, testData, testLabels):
	classifier = sklearn.naive_bayes.MultinomialNB()
	classifier.fit(trainData,trainLabels)
	guessedLabels = classifier.predict(testData)
	return numpy.mean(guessedLabels == testLabels)


def gaussNBTrainAndClassify(traindata, trainLabels, testData, testLabels):
	classifier = sklearn.naive_bayes.GaussianNB()
	classifier.fit(trainData,trainLabels)
	guessedLabels = classifier.predict(testData)
	return numpy.mean(guessedLabels == testLabels)
import numpy
import nltk
import scipy
from scipy import sparse
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn

#Import functions from other documents
import featureExtraction
import featureSelection
from classifiers import svmTrainAndClassify
from classifiers import multNBTrainAndClassify
from classifiers import gaussNBTrainAndClassify

#Script that imports data, extracts features, selects features, creates classifiers, and analyzes the results


############################################
###Import Raw Data##########################
############################################

#Create bunch from trainingData
trainBunch = sklearn.datasets.load_files("./c50/c50train/",load_content=True,encoding="utf-8",shuffle=False)

#Create bunch from testData
testBunch = sklearn.datasets.load_files("./c50/c50test/",load_content=True,encoding="utf-8",shuffle=False)






############################################
###Extract Features#########################
###And Split Training, Test, & Validation###
############################################
trainLabels = trainBunch.get('target')
trainData = numpy.array([0,0,0,0,0])
for document in trainBunch.get('data'):
	#CALL TO ISHAN'S CODE HERE
	#print temp
	trainData = numpy.vstack((trainData,temp))
trainData = numpy.delete(trainData, (0), axis=0)
#print trainData

ValidationLabels = testBunch.get('target')
ValidationData = numpy.array([0,0,0,0,0])
for document in testBunch.get('data'):
	#CALL TO ISHAN'S CODE HERE
	#print temp
	ValidationData = numpy.vstack((ValidationData,temp))
ValidationData = numpy.delete(ValidationData, (0), axis=0)
#print ValidationData

#ISHAN'S CODE FOR BAG OF WORDS
#print (trainDataTfidf.shape);





#Merge all separate matricies
print trainDataTfidf.shape
print trainData.shape
trainData = sparse.hstack((trainData,trainDataTfidf))
ValidationData = sparse.hstack((ValidationData,ValidationDataTfidf))
trainData = trainData.toarray()
ValidationData = ValidationData.toarray()

#Scale data
scaler = preprocessing.StandardScaler(with_mean=False).fit(trainData)
trainData = scaler.transform(trainData)
ValidationData = scaler.transform(ValidationData)

# minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(trainData)
# trainDataNB = minMaxScaler.transform(trainData)
# ValidationDataNB = minMaxScaler.transform(ValidationData)


############################################
###Manually Select Features#################
############################################









############################################
###Automatically Select Features############
############################################

#Generate percents based on size of total features
tenPercent = int(0.1 * trainData.shape[1])
twentyFivePercent = int(0.25 * trainData.shape[1])
fiftyPercent = int(0.5 * trainData.shape[1])



#Generate PCA models that reduce dimension to 10%, 25%, and 50% of initial size
pca10Model = featureSelection.pcaDecompTrain(trainData,tenPercent)
pca25Model = featureSelection.pcaDecompTrain(trainData,twentyFivePercent)
pca50Model = featureSelection.pcaDecompTrain(trainData,fiftyPercent)

#Generate Feature Agglomeration models that reduce dimension to 10%, 25%, and 50% of initial size
agglom10Model = featureSelection.agglomTrain(trainData,tenPercent)
agglom25Model = featureSelection.agglomTrain(trainData,twentyFivePercent)
agglom50Model = featureSelection.agglomTrain(trainData,fiftyPercent)

#Generate gaussian random projection models that reduce dimension to 10%, 25%, 50%, and auto of initial size
gaussRand10Model = featureSelection.gaussianRandProjTrain(trainData,tenPercent)
gaussRand25Model = featureSelection.gaussianRandProjTrain(trainData,twentyFivePercent)
gaussRand50Model = featureSelection.gaussianRandProjTrain(trainData,fiftyPercent)
#gaussRandAutoModel = featureSelection.gaussianRandProjTrain(trainData,"auto")

#Generate sparse random projection models that reduce dimension to 10%, 25%, 50%, and auto of initial size
sparseRand10Model = featureSelection.sparseRandProjTrain(trainData,tenPercent)
sparseRand25Model = featureSelection.sparseRandProjTrain(trainData,twentyFivePercent)
sparseRand50Model = featureSelection.sparseRandProjTrain(trainData,fiftyPercent)
#sparseRandAutoModel = featureSelection.sparseRandProjTrain(trainData,"auto")

#Reduce features based on models created above
pcaTrainData10 = featureSelection.featureReduce(pca10Model,trainData)
pcaValidationData10 = featureSelection.featureReduce(pca10Model,ValidationData)
pcaTrainData25 = featureSelection.featureReduce(pca25Model,trainData)
pcaValidationData25 = featureSelection.featureReduce(pca25Model,ValidationData)
pcaTrainData50 = featureSelection.featureReduce(pca50Model,trainData)
pcaValidationData50 = featureSelection.featureReduce(pca50Model,ValidationData)

agglomTrainData10 = featureSelection.featureReduce(agglom10Model,trainData)
agglomValidationData10 = featureSelection.featureReduce(agglom10Model,ValidationData)
agglomTrainData25 = featureSelection.featureReduce(agglom25Model,trainData)
agglomValidationData25 = featureSelection.featureReduce(agglom25Model,ValidationData)
agglomTrainData50 = featureSelection.featureReduce(agglom50Model,trainData)
agglomValidationData50 = featureSelection.featureReduce(agglom50Model,ValidationData)

gaussRandTrainData10 = featureSelection.featureReduce(gaussRand10Model,trainData)
gaussRandValidationData10 = featureSelection.featureReduce(gaussRand10Model,ValidationData)
gaussRandTrainData25 = featureSelection.featureReduce(gaussRand25Model,trainData)
gaussRandValidationData25 = featureSelection.featureReduce(gaussRand25Model,ValidationData)
gaussRandTrainData50 = featureSelection.featureReduce(gaussRand50Model,trainData)
gaussRandValidationData50 = featureSelection.featureReduce(gaussRand50Model,ValidationData)
#gaussRandTrainDataAuto = featureSelection.featureReduce(gaussRandAutoModel,trainData)
#gaussRandValidationDataAuto = featureSelection.featureReduce(gaussRandAutoModel,ValidationData)

sparseRandTrainData10 = featureSelection.featureReduce(sparseRand10Model,trainData)
sparseRandValidationData10 = featureSelection.featureReduce(sparseRand10Model,ValidationData)
sparseRandTrainData25 = featureSelection.featureReduce(sparseRand25Model,trainData)
sparseRandValidationData25 = featureSelection.featureReduce(sparseRand25Model,ValidationData)
sparseRandTrainData50 = featureSelection.featureReduce(sparseRand50Model,trainData)
sparseRandValidationData50 = featureSelection.featureReduce(sparseRand50Model,ValidationData)
#sparseRandTrainDataAuto = featureSelection.featureReduce(sparseRandAutoModel,trainData)
#sparseRandValidationDataAuto = featureSelection.featureReduce(sparseRandAutoModel,ValidationData)


#Generate only positive features for Naive Bayes classifiers
minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(pcaTrainData10)
pcaTrainData10NB = minMaxScaler.transform(pcaTrainData10)
pcaValidationData10NB = minMaxScaler.transform(pcaValidationData10)

minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(pcaTrainData25)
pcaTrainData25NB = minMaxScaler.transform(pcaTrainData25)
pcaValidationData25NB = minMaxScaler.transform(pcaValidationData25)

minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(pcaTrainData50)
pcaTrainData50NB = minMaxScaler.transform(pcaTrainData50)
pcaValidationData50NB = minMaxScaler.transform(pcaValidationData50)


minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(agglomTrainData10)
agglomTrainData10NB = minMaxScaler.transform(agglomTrainData10)
agglomValidationData10NB = minMaxScaler.transform(agglomValidationData10)

minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(agglomTrainData25)
agglomTrainData25NB = minMaxScaler.transform(agglomTrainData25)
agglomValidationData25NB = minMaxScaler.transform(agglomValidationData25)

minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(agglomTrainData50)
agglomTrainData50NB = minMaxScaler.transform(agglomTrainData50)
agglomValidationData50NB = minMaxScaler.transform(agglomValidationData50)


minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(gaussRandTrainData10)
gaussRandTrainData10NB = minMaxScaler.transform(gaussRandTrainData10)
gaussRandValidationData10NB = minMaxScaler.transform(gaussRandValidationData10)

minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(gaussRandTrainData25)
gaussRandTrainData25NB = minMaxScaler.transform(gaussRandTrainData25)
gaussRandValidationData25NB = minMaxScaler.transform(gaussRandValidationData25)

minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(gaussRandTrainData50)
gaussRandTrainData50NB = minMaxScaler.transform(gaussRandTrainData50)
gaussRandValidationData50NB = minMaxScaler.transform(gaussRandValidationData50)


minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(sparseRandTrainData10)
sparseRandTrainData10NB = minMaxScaler.transform(sparseRandTrainData10)
sparseRandValidationData10NB = minMaxScaler.transform(sparseRandValidationData10)

minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(sparseRandTrainData25)
sparseRandTrainData25NB = minMaxScaler.transform(sparseRandTrainData25)
sparseRandValidationData25NB = minMaxScaler.transform(sparseRandValidationData25)

minMaxScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1)).fit(sparseRandTrainData50)
sparseRandTrainData50NB = minMaxScaler.transform(sparseRandTrainData50)
sparseRandValidationData50NB = minMaxScaler.transform(sparseRandValidationData50)

###########################################
##Run Classifiers on Validation############
###########################################

#run classifiers with varying parameters using train & validation data from feature selection sections

#For SVM:
#cVal can be 0.1,1,10
#kernelVal can be linear, poly2, poly3, rbf

#linear c=0.1
print "linear SVM, C=0.1   ################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0.1, "linear",4)
#print accuracy
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 0.1, "linear", 4)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 0.1, "linear", 4)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 0.1, "linear", 4)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 0.1, "linear", 4)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 0.1, "linear", 4)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 0.1, "linear", 4)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 0.1, "linear", 4)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 0.1, "linear", 4)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 0.1, "linear", 4)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 0.1, "linear", 4)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 0.1, "linear", 4)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 0.1, "linear", 4)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 0.1, "linear", 4)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


#linear c=1
print "linear SVM, C=1   ##################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 1, "linear", 4)
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 1, "linear", 4)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 1, "linear", 4)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 1, "linear", 4)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 1, "linear", 4)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 1, "linear", 4)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 1, "linear", 4)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 1, "linear", 4)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 1, "linear", 4)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 1, "linear", 4)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 1, "linear", 4)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 1, "linear", 4)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 1, "linear", 4)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 1, "linear", 4)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


#linear c=10
print "linear SVM, C=10   #################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 10, "linear", 4)
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 10, "linear", 4)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 10, "linear", 4)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 10, "linear", 4)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 10, "linear", 4)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 10, "linear", 4)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 10, "linear", 4)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 10, "linear", 4)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 10, "linear", 4)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 10, "linear", 4)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 10, "linear", 4)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 10, "linear", 4)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 10, "linear", 4)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 10, "linear", 4)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


#Poly 2 c=0.1
print "Poly 2 SVM, C=0.1   ################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0.1, "poly", 2)
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 0.1, "poly", 2)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 0.1, "poly", 2)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 0.1, "poly", 2)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 0.1, "poly", 2)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 0.1, "poly", 2)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 0.1, "poly", 2)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 0.1, "poly", 2)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 0.1, "poly", 2)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 0.1, "poly", 2)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 0.1, "poly", 2)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 0.1, "poly", 2)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 0.1, "poly", 2)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 0.1, "poly", 2)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


#Poly 2 c=1
print "Poly 2 SVM, C=1   ##################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 1, "poly", 2)
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 1, "poly", 2)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 1, "poly", 2)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 1, "poly", 2)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 1, "poly", 2)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 1, "poly", 2)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 1, "poly", 2)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 1, "poly", 2)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 1, "poly", 2)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 1, "poly", 2)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 1, "poly", 2)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 1, "poly", 2)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 1, "poly", 2)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 1, "poly", 2)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


#Poly 2 c=10
print "Poly 2 SVM, C=10   #################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 10, "poly", 2)
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 10, "poly", 2)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 10, "poly", 2)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 10, "poly", 2)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 10, "poly", 2)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 10, "poly", 2)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 10, "poly", 2)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 10, "poly", 2)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 10, "poly", 2)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 10, "poly", 2)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 10, "poly", 2)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 10, "poly", 2)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 10, "poly", 2)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 10, "poly", 2)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


# #Poly 3 c=0.1
# print "Poly 3 SVM, C=0.1   ################################"

# accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0.1, "poly", 3)
# print "PCA 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 0.1, "poly", 3)
# print "PCA 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 0.1, "poly", 3)
# print "PCA 50: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 0.1, "poly", 3)
# print "Feature Agglomeration 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 0.1, "poly", 3)
# print "Feature Agglomeration 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 0.1, "poly", 3)
# print "Feature Agglomeration 50: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 0.1, "poly", 3)
# print "Gaussian Random Tree 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 0.1, "poly", 3)
# print "Gaussian Random Tree 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 0.1, "poly", 3)
# print "Gaussian Random Tree 50: %0.5f" % (accuracy)
# #accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 0.1, "poly", 3)
# #print "Gaussian Random Tree auto: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 0.1, "poly", 3)
# print "Sparse Random Tree 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 0.1, "poly", 3)
# print "Sparse Random Tree 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 0.1, "poly", 3)
# print "Sparse Random Tree 50: %0.5f" % (accuracy)
# #accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 0.1, "poly", 3)
# #print "Sparse Random Tree auto: %0.5f" % (accuracy)


# #Poly 3 c=1
# print "Poly 3 SVM, C=1   ##################################"

# accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 1, "poly", 3)
# print "PCA 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 1, "poly", 3)
# print "PCA 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 1, "poly", 3)
# print "PCA 50: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 1, "poly", 3)
# print "Feature Agglomeration 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 1, "poly", 3)
# print "Feature Agglomeration 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 1, "poly", 3)
# print "Feature Agglomeration 50: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 1, "poly", 3)
# print "Gaussian Random Tree 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 1, "poly", 3)
# print "Gaussian Random Tree 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 1, "poly", 3)
# print "Gaussian Random Tree 50: %0.5f" % (accuracy)
# #accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 1, "poly", 3)
# #print "Gaussian Random Tree auto: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 1, "poly", 3)
# print "Sparse Random Tree 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 1, "poly", 3)
# print "Sparse Random Tree 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 1, "poly", 3)
# print "Sparse Random Tree 50: %0.5f" % (accuracy)
# #accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 1, "poly", 3)
# #print "Sparse Random Tree auto: %0.5f" % (accuracy)


# #Poly 3 c=10
# print "Poly 3 SVM, C=10   #################################"

# accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 10, "poly", 3)
# print "PCA 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 10, "poly", 3)
# print "PCA 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 10, "poly", 3)
# print "PCA 50: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 10, "poly", 3)
# print "Feature Agglomeration 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 10, "poly", 3)
# print "Feature Agglomeration 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 10, "poly", 3)
# print "Feature Agglomeration 50: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 10, "poly", 3)
# print "Gaussian Random Tree 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 10, "poly", 3)
# print "Gaussian Random Tree 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 10, "poly", 3)
# print "Gaussian Random Tree 50: %0.5f" % (accuracy)
# #accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 10, "poly", 3)
# #print "Gaussian Random Tree auto: %0.5f" % (accuracy)

# accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 10, "poly", 3)
# print "Sparse Random Tree 10: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 10, "poly", 3)
# print "Sparse Random Tree 25: %0.5f" % (accuracy)
# accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 10, "poly", 3)
# print "Sparse Random Tree 50: %0.5f" % (accuracy)
# #accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 10, "poly", 3)
# #print "Sparse Random Tree auto: %0.5f" % (accuracy)



#Rbf c=0.1
print "Rbf SVM, C=0.1   ###################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0.1, "rbf", 4)
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 0.1, "rbf", 4)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 0.1, "rbf", 4)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 0.1, "rbf", 4)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 0.1, "rbf", 4)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 0.1, "rbf", 4)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 0.1, "rbf", 4)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 0.1, "rbf", 4)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 0.1, "rbf", 4)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 0.1, "rbf", 4)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 0.1, "rbf", 4)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 0.1, "rbf", 4)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 0.1, "rbf", 4)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 0.1, "rbf", 4)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


#Rbf c=1
print "Rbf SVM, C=1   #####################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 1, "rbf", 4)
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 1, "rbf", 4)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 1, "rbf", 4)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 1, "rbf", 4)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 1, "rbf", 4)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 1, "rbf", 4)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 1, "rbf", 4)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 1, "rbf", 4)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 1, "rbf", 4)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 1, "rbf", 4)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 1, "rbf", 4)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 1, "rbf", 4)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 1, "rbf", 4)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 1, "rbf", 4)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


#Rbf c=10
print "Rbf SVM, C=10   ####################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 10, "rbf", 4)
print "PCA 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 10, "rbf", 4)
print "PCA 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 10, "rbf", 4)
print "PCA 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 10, "rbf", 4)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 10, "rbf", 4)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 10, "rbf", 4)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 10, "rbf", 4)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 10, "rbf", 4)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 10, "rbf", 4)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 10, "rbf", 4)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 10, "rbf", 4)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 10, "rbf", 4)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 10, "rbf", 4)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 10, "rbf", 4)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)





#For Naive Bayes:
#multinomial & guassian

#Multinomial
print "Multinomial Naive Bayes   ##########################"

accuracy = multNBTrainAndClassify(pcaTrainData10NB, trainLabels, pcaValidationData10NB, ValidationLabels)
print "PCA 10: %0.5f" % (accuracy)
accuracy = multNBTrainAndClassify(pcaTrainData25NB, trainLabels, pcaValidationData25NB, ValidationLabels)
print "PCA 25: %0.5f" % (accuracy)
accuracy = multNBTrainAndClassify(pcaTrainData50NB, trainLabels, pcaValidationData50NB, ValidationLabels)
print "PCA 50: %0.5f" % (accuracy)

accuracy = multNBTrainAndClassify(agglomTrainData10NB, trainLabels, agglomValidationData10NB, ValidationLabels)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = multNBTrainAndClassify(agglomTrainData25NB, trainLabels, agglomValidationData25NB, ValidationLabels)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = multNBTrainAndClassify(agglomTrainData50NB, trainLabels, agglomValidationData50NB, ValidationLabels)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = multNBTrainAndClassify(gaussRandTrainData10NB, trainLabels, gaussRandValidationData10NB, ValidationLabels)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = multNBTrainAndClassify(gaussRandTrainData25NB, trainLabels, gaussRandValidationData25NB, ValidationLabels)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = multNBTrainAndClassify(gaussRandTrainData50NB, trainLabels, gaussRandValidationData50NB, ValidationLabels)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = multNBTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = multNBTrainAndClassify(sparseRandTrainData10NB, trainLabels, sparseRandValidationData10NB, ValidationLabels)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = multNBTrainAndClassify(sparseRandTrainData25NB, trainLabels, sparseRandValidationData25NB, ValidationLabels)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = multNBTrainAndClassify(sparseRandTrainData50NB, trainLabels, sparseRandValidationData50NB, ValidationLabels)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = multNBTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


#Gaussian
print "Gaussian Naive Bayes   #############################"

accuracy = gaussNBTrainAndClassify(pcaTrainData10NB, trainLabels, pcaValidationData10NB, ValidationLabels)
print "PCA 10: %0.5f" % (accuracy)
accuracy = gaussNBTrainAndClassify(pcaTrainData25NB, trainLabels, pcaValidationData25NB, ValidationLabels)
print "PCA 25: %0.5f" % (accuracy)
accuracy = gaussNBTrainAndClassify(pcaTrainData50NB, trainLabels, pcaValidationData50NB, ValidationLabels)
print "PCA 50: %0.5f" % (accuracy)

accuracy = gaussNBTrainAndClassify(agglomTrainData10NB, trainLabels, agglomValidationData10NB, ValidationLabels)
print "Feature Agglomeration 10: %0.5f" % (accuracy)
accuracy = gaussNBTrainAndClassify(agglomTrainData25NB, trainLabels, agglomValidationData25NB, ValidationLabels)
print "Feature Agglomeration 25: %0.5f" % (accuracy)
accuracy = gaussNBTrainAndClassify(agglomTrainData50NB, trainLabels, agglomValidationData50NB, ValidationLabels)
print "Feature Agglomeration 50: %0.5f" % (accuracy)

accuracy = gaussNBTrainAndClassify(gaussRandTrainData10NB, trainLabels, gaussRandValidationData10NB, ValidationLabels)
print "Gaussian Random Tree 10: %0.5f" % (accuracy)
accuracy = gaussNBTrainAndClassify(gaussRandTrainData25NB, trainLabels, gaussRandValidationData25NB, ValidationLabels)
print "Gaussian Random Tree 25: %0.5f" % (accuracy)
accuracy = gaussNBTrainAndClassify(gaussRandTrainData50NB, trainLabels, gaussRandValidationData50NB, ValidationLabels)
print "Gaussian Random Tree 50: %0.5f" % (accuracy)
#accuracy = gaussNBTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels)
#print "Gaussian Random Tree auto: %0.5f" % (accuracy)

accuracy = gaussNBTrainAndClassify(sparseRandTrainData10NB, trainLabels, sparseRandValidationData10NB, ValidationLabels)
print "Sparse Random Tree 10: %0.5f" % (accuracy)
accuracy = gaussNBTrainAndClassify(sparseRandTrainData25NB, trainLabels, sparseRandValidationData25NB, ValidationLabels)
print "Sparse Random Tree 25: %0.5f" % (accuracy)
accuracy = gaussNBTrainAndClassify(sparseRandTrainData50NB, trainLabels, sparseRandValidationData50NB, ValidationLabels)
print "Sparse Random Tree 50: %0.5f" % (accuracy)
#accuracy = gaussNBTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels)
#print "Sparse Random Tree auto: %0.5f" % (accuracy)


############################################
###Run Best Classifier on Test##############
############################################

#This code was determined after running the above code once to see the highest performing classifiers on validation data

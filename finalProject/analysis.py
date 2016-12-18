import numpy
import nltk
import scipy
import sklearn

#Import functions from other documents
import featureSelection
import classifiers

#Script that imports data, extracts features, selects features, creates classifiers, and analyzes the results


############################################
###Import Raw Data##########################
############################################









############################################
###Extract Features#########################
###And Split Training, Test, & Validation###
############################################









############################################
###Manually Select Features#################
############################################









############################################
###Automatically Select Features############
############################################

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
gaussRandAutoModel = featureSelection.gaussianRandProjTrain(trainData,"auto")

#Generate sparse random projection models that reduce dimension to 10%, 25%, 50%, and auto of initial size
sparseRand10Model = featureSelection.sparseRandProjTrain(trainData,tenPercent)
sparseRand25Model = featureSelection.sparseRandProjTrain(trainData,twentyFivePercent)
sparseRand50Model = featureSelection.sparseRandProjTrain(trainData,fiftyPercent)
sparseRandAutoModel = featureSelection.sparseRandProjTrain(trainData,"auto")

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
gaussRandTrainDataAuto = featureSelection.featureReduce(gaussRandAutoModel,trainData)
gaussRandValidationDataAuto = featureSelection.featureReduce(gaussRandAutoModel,ValidationData)

sparseRandTrainData10 = featureSelection.featureReduce(sparseRand10Model,trainData)
sparseRandValidationData10 = featureSelection.featureReduce(sparseRand10Model,ValidationData)
sparseRandTrainData25 = featureSelection.featureReduce(sparseRand25Model,trainData)
sparseRandValidationData25 = featureSelection.featureReduce(sparseRand25Model,ValidationData)
sparseRandTrainData50 = featureSelection.featureReduce(sparseRand50Model,trainData)
sparseRandValidationData50 = featureSelection.featureReduce(sparseRand50Model,ValidationData)
sparseRandTrainDataAuto = featureSelection.featureReduce(sparseRandAutoModel,trainData)
sparseRandValidationDataAuto = featureSelection.featureReduce(sparseRandAutoModel,ValidationData)



############################################
###Run Classifiers on Validation############
############################################

#run classifiers with varying parameters using train & validation data from feature selection sections

#For SVM:
#cVal can be 0.1,1,10
#kernelVal can be linear, poly2, poly3, rbf

#linear c=0.1
print "linear SVM, C=0.1   ################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0.1, "linear", 4)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 0.1, "linear", 4)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 0.1, "linear", 4)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 0.1, "linear", 4)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 0.1, "linear", 4)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 0.1, "linear", 4)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 0.1, "linear", 4)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 0.1, "linear", 4)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 0.1, "linear", 4)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 0.1, "linear", 4)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 0.1, "linear", 4)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 0.1, "linear", 4)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 0.1, "linear", 4)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 0.1, "linear", 4)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#linear c=1
print "linear SVM, C=1   ##################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 1, "linear", 4)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 1, "linear", 4)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 1, "linear", 4)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 1, "linear", 4)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 1, "linear", 4)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 1, "linear", 4)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 1, "linear", 4)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 1, "linear", 4)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 1, "linear", 4)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 1, "linear", 4)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 1, "linear", 4)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 1, "linear", 4)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 1, "linear", 4)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 1, "linear", 4)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#linear c=10
print "linear SVM, C=10   #################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 10, "linear", 4)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 10, "linear", 4)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 10, "linear", 4)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 10, "linear", 4)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 10, "linear", 4)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 10, "linear", 4)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 10, "linear", 4)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 10, "linear", 4)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 10, "linear", 4)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 10, "linear", 4)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 10, "linear", 4)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 10, "linear", 4)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 10, "linear", 4)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 10, "linear", 4)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Poly 2 c=0.1
print "Poly 2 SVM, C=0.1   ################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0.1, "poly", 2)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 0.1, "poly", 2)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 0.1, "poly", 2)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 0.1, "poly", 2)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 0.1, "poly", 2)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 0.1, "poly", 2)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 0.1, "poly", 2)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 0.1, "poly", 2)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 0.1, "poly", 2)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 0.1, "poly", 2)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 0.1, "poly", 2)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 0.1, "poly", 2)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 0.1, "poly", 2)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 0.1, "poly", 2)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Poly 2 c=1
print "Poly 2 SVM, C=1   ##################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 1, "poly", 2)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 1, "poly", 2)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 1, "poly", 2)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 1, "poly", 2)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 1, "poly", 2)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 1, "poly", 2)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 1, "poly", 2)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 1, "poly", 2)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 1, "poly", 2)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 1, "poly", 2)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 1, "poly", 2)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 1, "poly", 2)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 1, "poly", 2)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 1, "poly", 2)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Poly 2 c=10
print "Poly 2 SVM, C=10   #################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0, "poly", 2)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 10, "poly", 2)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 10, "poly", 2)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 10, "poly", 2)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 10, "poly", 2)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 10, "poly", 2)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 10, "poly", 2)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 10, "poly", 2)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 10, "poly", 2)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 10, "poly", 2)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 10, "poly", 2)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 10, "poly", 2)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 10, "poly", 2)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 10, "poly", 2)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Poly 3 c=0.1
print "Poly 3 SVM, C=0.1   ################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0.1, "poly", 3)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 0.1, "poly", 3)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 0.1, "poly", 3)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 0.1, "poly", 3)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 0.1, "poly", 3)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 0.1, "poly", 3)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 0.1, "poly", 3)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 0.1, "poly", 3)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 0.1, "poly", 3)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 0.1, "poly", 3)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 0.1, "poly", 3)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 0.1, "poly", 3)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 0.1, "poly", 3)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 0.1, "poly", 3)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Poly 3 c=1
print "Poly 3 SVM, C=1   ##################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 1, "poly", 3)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 1, "poly", 3)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 1, "poly", 3)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 1, "poly", 3)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 1, "poly", 3)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 1, "poly", 3)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 1, "poly", 3)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 1, "poly", 3)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 1, "poly", 3)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 1, "poly", 3)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 1, "poly", 3)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 1, "poly", 3)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 1, "poly", 3)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 1, "poly", 3)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Poly 3 c=10
print "Poly 3 SVM, C=10   #################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 10, "poly", 3)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 10, "poly", 3)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 10, "poly", 3)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 10, "poly", 3)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 10, "poly", 3)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 10, "poly", 3)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 10, "poly", 3)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 10, "poly", 3)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 10, "poly", 3)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 10, "poly", 3)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 10, "poly", 3)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 10, "poly", 3)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 10, "poly", 3)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 10, "poly", 3)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Rbf c=0.1
print "Rbf SVM, C=0.1   ###################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 0.1, "rbf", 4)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 0.1, "rbf", 4)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 0.1, "rbf", 4)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 0.1, "rbf", 4)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 0.1, "rbf", 4)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 0.1, "rbf", 4)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 0.1, "rbf", 4)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 0.1, "rbf", 4)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 0.1, "rbf", 4)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 0.1, "rbf", 4)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 0.1, "rbf", 4)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 0.1, "rbf", 4)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 0.1, "rbf", 4)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 0.1, "rbf", 4)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Rbf c=1
print "Rbf SVM, C=1   #####################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 1, "rbf", 4)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 1, "rbf", 4)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 1, "rbf", 4)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 1, "rbf", 4)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 1, "rbf", 4)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 1, "rbf", 4)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 1, "rbf", 4)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 1, "rbf", 4)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 1, "rbf", 4)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 1, "rbf", 4)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 1, "rbf", 4)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 1, "rbf", 4)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 1, "rbf", 4)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 1, "rbf", 4)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Rbf c=10
print "Rbf SVM, C=10   ####################################"

accuracy = svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, 10, "rbf", 4)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels, 10, "rbf", 4)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels, 10, "rbf", 4)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels, 10, "rbf", 4)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels, 10, "rbf", 4)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels, 10, "rbf", 4)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels, 10, "rbf", 4)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels, 10, "rbf", 4)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels, 10, "rbf", 4)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels, 10, "rbf", 4)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = svmTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels, 10, "rbf", 4)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels, 10, "rbf", 4)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels, 10, "rbf", 4)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = svmTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels, 10, "rbf", 4)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)



#For Naive Bayes:
#multinomial & guassian

#Multinomial
print "Multinomial Naive Bayes   ##########################"

accuracy = multNBTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = multNBTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = multNBTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = multNBTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = multNBTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


#Gaussian
print "Gaussian Naive Bayes   #############################"

accuracy = gaussNBTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels)
print "PCA 10: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(pcaTrainData25, trainLabels, pcaValidationData25, ValidationLabels)
print "PCA 25: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(pcaTrainData50, trainLabels, pcaValidationData50, ValidationLabels)
print "PCA 50: %0.5f%" % (accuracy)

accuracy = gaussNBTrainAndClassify(agglomTrainData10, trainLabels, agglomValidationData10, ValidationLabels)
print "Feature Agglomeration 10: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(agglomTrainData25, trainLabels, agglomValidationData25, ValidationLabels)
print "Feature Agglomeration 25: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(agglomTrainData50, trainLabels, agglomValidationData50, ValidationLabels)
print "Feature Agglomeration 50: %0.5f%" % (accuracy)

accuracy = gaussNBTrainAndClassify(gaussRandTrainData10, trainLabels, gaussRandValidationData10, ValidationLabels)
print "Gaussian Random Tree 10: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(gaussRandTrainData25, trainLabels, gaussRandValidationData25, ValidationLabels)
print "Gaussian Random Tree 25: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(gaussRandTrainData50, trainLabels, gaussRandValidationData50, ValidationLabels)
print "Gaussian Random Tree 50: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(gaussRandTrainDataAuto, trainLabels, gaussRandValidationDataAuto, ValidationLabels)
print "Gaussian Random Tree auto: %0.5f%" % (accuracy)

accuracy = gaussNBTrainAndClassify(sparseRandTrainData10, trainLabels, sparseRandValidationData10, ValidationLabels)
print "Sparse Random Tree 10: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(sparseRandTrainData25, trainLabels, sparseRandValidationData25, ValidationLabels)
print "Sparse Random Tree 25: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(sparseRandTrainData50, trainLabels, sparseRandValidationData50, ValidationLabels)
print "Sparse Random Tree 50: %0.5f%" % (accuracy)
accuracy = gaussNBTrainAndClassify(sparseRandTrainDataAuto, trainLabels, sparseRandValidationDataAuto, ValidationLabels)
print "Sparse Random Tree auto: %0.5f%" % (accuracy)


############################################
###Run Best Classifier on Test##############
############################################

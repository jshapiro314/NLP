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
pcaTestData10 = featureSelection.featureReduce(pca10Model,testData)
pcaTrainData25 = featureSelection.featureReduce(pca25Model,trainData)
pcaTestData25 = featureSelection.featureReduce(pca25Model,testData)
pcaTrainData50 = featureSelection.featureReduce(pca50Model,trainData)
pcaTestData50 = featureSelection.featureReduce(pca50Model,testData)

agglomTrainData10 = featureSelection.featureReduce(agglom10Model,trainData)
agglomTestData10 = featureSelection.featureReduce(agglom10Model,testData)
agglomTrainData25 = featureSelection.featureReduce(agglom25Model,trainData)
agglomTestData25 = featureSelection.featureReduce(agglom25Model,testData)
agglomTrainData50 = featureSelection.featureReduce(agglom50Model,trainData)
agglomTestData50 = featureSelection.featureReduce(agglom50Model,testData)

gaussRandTrainData10 = featureSelection.featureReduce(gaussRand10Model,trainData)
gaussRandTestData10 = featureSelection.featureReduce(gaussRand10Model,testData)
gaussRandTrainData25 = featureSelection.featureReduce(gaussRand25Model,trainData)
gaussRandTestData25 = featureSelection.featureReduce(gaussRand25Model,testData)
gaussRandTrainData50 = featureSelection.featureReduce(gaussRand50Model,trainData)
gaussRandTestData50 = featureSelection.featureReduce(gaussRand50Model,testData)
gaussRandTrainDataAuto = featureSelection.featureReduce(gaussRandAutoModel,trainData)
gaussRandTestDataAuto = featureSelection.featureReduce(gaussRandAutoModel,testData)

sparseRandTrainData10 = featureSelection.featureReduce(sparseRand10Model,trainData)
sparseRandTestData10 = featureSelection.featureReduce(sparseRand10Model,testData)
sparseRandTrainData25 = featureSelection.featureReduce(sparseRand25Model,trainData)
sparseRandTestData25 = featureSelection.featureReduce(sparseRand25Model,testData)
sparseRandTrainData50 = featureSelection.featureReduce(sparseRand50Model,trainData)
sparseRandTestData50 = featureSelection.featureReduce(sparseRand50Model,testData)
sparseRandTrainDataAuto = featureSelection.featureReduce(sparseRandAutoModel,trainData)
sparseRandTestDataAuto = featureSelection.featureReduce(sparseRandAutoModel,testData)



############################################
###Run Classifiers##########################
############################################

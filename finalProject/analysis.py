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
svmTrainAndClassify(pcaTrainData10, trainLabels, pcaValidationData10, ValidationLabels, cVal, kernelVal, degreeVal)






############################################
###Run Best Classifier on Test##############
############################################

import numpy
import nltk
import scipy
import sklearn

#Methods that handle automated feature reduction
#Methods that end in train generate reduction models
#Method that ends in reduce reduces features based on input model



def pcaDecompTrain(trainData,numFeatures):
	pcaModel = sklearn.decomposiiton.PCA(n_components = numFeatures)
	pcaModel.fit(trainData)
	return pcaModel


def agglomTrain(trainData,numFeatures):
	agglomModel = sklearn.cluster.FeatureAgglomeration(n_clusters = numFeatures)
	agglomModel.fit(trainData)
	return agglomModel


#Can take 'auto' as numFeatuers
def gaussianRandProjTrain(trainData,numFeatures):
	gaussModel = sklearn.random_projection.GaussianRandomProjection(n_components = numFeatures)
	gaussModel.fit(trainData)
	return gaussModel

#Can take 'auto' as numFeatuers
def sparseRandProjTrain(trainData,numFeatures):
	sparseModel = sklearn.random_projection.SparseRandomProjection(n_components = numFeatures)
	sparseModel.fit(trainData)
	return gaussModel

def featureReduce(selectionModel,data):
	return selectionModel.transform(data)


import numpy
import nltk
import scipy
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
#Methods that handle automated feature reduction
#Methods that end in train generate reduction models
#Method that ends in reduce reduces features based on input model



def pcaDecompTrain(trainData,numFeatures):
	pcaModel = PCA(n_components = numFeatures)
	pcaModel.fit(trainData)
	return pcaModel


def agglomTrain(trainData,numFeatures):
	agglomModel = FeatureAgglomeration(n_clusters = numFeatures)
	agglomModel.fit(trainData)
	return agglomModel


#Can take 'auto' as numFeatuers
def gaussianRandProjTrain(trainData,numFeatures):
	gaussModel = GaussianRandomProjection(n_components = numFeatures)
	gaussModel.fit(trainData)
	return gaussModel

#Can take 'auto' as numFeatuers
def sparseRandProjTrain(trainData,numFeatures):
	sparseModel = SparseRandomProjection(n_components = numFeatures)
	sparseModel.fit(trainData)
	return sparseModel

def featureReduce(selectionModel,data):
	return selectionModel.transform(data)


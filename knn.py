import numpy as np
from scipy.io import loadmat
import csv
from numpy.matlib import repmat

training = "train.mat"
testing = "test.mat"

training_data = loadmat(training)
testing_data = loadmat(testing)
xTr = training_data["x"]
yTr = np.round(training_data["y"])
yTr = np.argmax(yTr, axis = 1)
xTe = testing_data["x"]


def l2distance(X, Z=None):
    """
    function D=l2distance(X,Z)

    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: dxn data matrix with n vectors (columns) of dimensionality d
    Z: dxm data matrix with m vectors (columns) of dimensionality d

    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)

    call with only one input:
    l2distance(X)=l2distance(X,X)
    """
    if Z is None:
        n, d = X.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        D1 = -2 * np.dot(X, X.T) + repmat(s1, 1, n)
        D = D1 + repmat(s1.T, n, 1)
        np.fill_diagonal(D, 0)
        D = np.sqrt(np.maximum(D, 0))
    else:
        n, d = X.shape
        m, _ = Z.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        s2 = np.sum(np.power(Z, 2), axis=1).reshape(1,-1)
        D1 = -2 * np.dot(X, Z.T) + repmat(s1, 1, m)
        D = D1 + repmat(s2, n, 1)
        D = np.sqrt(np.maximum(D, 0))
    return D

def findknn(xTr,xTe,k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);
    
    Finds the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """
    
    D = l2distance(xTr, xTe)
    indices = np.argsort(D, axis = 0)
   
    indices = indices[0:k,:]
    D.sort(axis = 0)
    dists = D[0:k,:]
    
   
    return indices, dists

def hasDuplicateMode(a):
    if np.size(a) == 1:
        return False
    else:
        return a[0] == a[1]

def findMode(predictions):
    temp = predictions
    pred = predictions.astype(int)
    temp = pred
    bins = np.bincount(pred)
    bins2 = np.sort(bins)[::-1]
    if hasDuplicateMode(bins2):    
        maxbin = findMode(temp[:-1])
    else:
        maxbin = np.argmax(bins)
    return np.array([maxbin])

def returnvalue(predictions):
    a = findMode(predictions)
    b = findMode(predictions[::-1])
    c = findMode(predictions[:int(predictions.size/2)+1])
    if a == c:
        return a
    else:
        return b

def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);

    k-nn classifier 

    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found

    Output:

    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    indices, dists = findknn(xTr, xTe, k)
  

    predictions = np.take(yTr, indices.T)
    modes = np.apply_along_axis(returnvalue, 1, predictions)


    xx = modes.T.flatten()
    return xx

if __name__ == "__main__":
	
	training_data = loadmat(training)
	testing_data = loadmat(testing)
	xTr = training_data["x"]
	yTr = np.round(training_data["y"])
	yTr = np.argmax(yTr, axis = 1)
	xTe = testing_data["x"]

	testpredictions = knnclassifier(xTr, yTr, xTe, 37)
	n,d = xTe.shape

	with open("results.csv", 'w') as csvfile:
		writer = csv.writer(csvfile)
 		writer.writerow(['id','digit'])
 		for i in range(n):
 			writer.writerow([i, testpredictions[i]])
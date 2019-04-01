import features_extraction
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

def normalizeFeatures(features , min=None , max=None):
    # means=np.sum(features, axis=0)
    # std=np.std(features,axis=0)
    if min is None:
        min = np.min(features, axis=0)
        max = np.max(features, axis=0)
        diff=max-min
        features=(features-min)/diff
    else :
        diff = max - min
        features = (features - min) / diff
    return features , min , max

# ##Extract Easy Features
# easyFeatures = english.extractFeatures("./Easy/*")
# easyLabels = np.zeros(easyFeatures.shape[0])
# print("finish easy")
# ##Extract Difficult Features
# difficultFeatures = english.extractFeatures("./Difficult/*")
# difficultLabels = np.ones(difficultFeatures.shape[0])
# print("finish difficult")
# ##Merge Features
# features = np.vstack((easyFeatures , difficultFeatures))
#
# ##features=normalizeFeatures(features)
# labels = np.concatenate((easyLabels , difficultLabels))
#
#
# print("------------------- To  file---------------------------")
#
# with open('features.csv', 'w') as writeFile:
#     writer = csv.writer(writeFile)
#     writer.writerows(features)
#
# with open('labels.txt', 'w') as f:
#     for item in labels:
#         f.write("%s\n" % item)



##Read features

#with open('features.csv', 'r') as readFile:
#    reader = csv.reader(readFile)
#    features = list(reader)
#features=np.array(features).astype('float32',casting='unsafe')

labels=np.array(open('labels.txt').read().split()).astype('float32' , casting='unsafe')
labels=labels.astype('uint32' , casting='unsafe')

features = pd.DataFrame(pd.read_csv('features.csv' , header=None))
features=features.apply(pd.to_numeric , errors='coerce')

# simpleIndices=np.where(labels==0)
# complexIndices=np.where(labels==1)
#
# for i in range(0,features.shape[1]):
#     x = np.array(features.iloc[:, i])
#     print("Column ", i, " : ", np.std(x))
#     for j in range(i+1 , features.shape[1]):
#         y = np.array(features.iloc[:, j])
#         print("Column ", j, " : ", np.std(y))
#         simplex=x[simpleIndices]
#         simpley=y[simpleIndices]
#         complexx=x[complexIndices]
#         complexy=y[complexIndices]
#         plt.scatter(simplex, simpley , c='b')
#         plt.scatter(complexx, complexy , c='r')
#         plt.show()



#
#
# print(features.head())
features= features.iloc[: ,  [0,1,5,6,8,9,12,13,14,15,16,17 ] ]
print(features.head())



##Scale Training
#scaler=StandardScaler()
#scaler.fit_transform(features)
# pca = PCA(n_components=12)
# pca.fit_transform(features)
# selector=SelectKBest(k=15 , score_func=f_regression)
# selector.fit_transform(features,labels)
features=np.array(features)
features , min , max =normalizeFeatures(features)

##Scale Testing
test = features_extraction.featuresFromFile("./test/test.txt")
test_labels=np.array(open('test/labels.txt').read().split()).astype('int')
test=pd.DataFrame(test)
test= test.iloc[: ,  [0,1,5,6,8,9,12,13,14,15,16,17 ]]
test=np.array(test)
test , a , c  =normalizeFeatures(test , min , max)

#scaler.transform(test)
#selector.transform(test)
#test=normalizeFeatures(test)

##Train Models
nnclf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nnclf.fit(features, labels)
print("NN prediction")
nnPrediction=nnclf.predict(test)
print(nnPrediction)


svmclf = svm.SVC(gamma='scale')
svmclf.fit(features, labels)
print("SVM Prediction")
svmPrediction=svmclf.predict(test)
print(svmPrediction)


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(features, labels)
print("KNN prediction")
knnPrediction=neigh.predict(test)
print(knnPrediction)

## Print results
with open('Results.txt', 'w') as f:
    f.write("NN \t KNN \t SVM")
    for i in range(len(nnPrediction)):
        f.write("\n %s" % nnPrediction[i])
        f.write("\t %s" % knnPrediction[i])
        f.write("\t %s" % svmPrediction[i])


## Compare Accuracies

print("NN ",np.sum(nnPrediction==test_labels)/len(test_labels))
print("KNN ",np.sum(knnPrediction==test_labels)/len(test_labels))
print("SVM ",np.sum(svmPrediction==test_labels)/len(test_labels))


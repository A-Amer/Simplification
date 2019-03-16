import english
from sklearn.neural_network import MLPClassifier
import numpy as np

## Extract Easy Features
easyFeatures = english.extractFeatures("./Easy/*")
easyLabels = np.zeros(easyFeatures.shape[0])
print("finish easy")
##Extract Difficult Features
difficultFeatures = english.extractFeatures("./Difficult/*")
difficultLabels = np.ones(difficultFeatures.shape[0])
print("finish difficult")


features = np.vstack((easyFeatures , difficultFeatures))
labels = np.concatenate((easyLabels , difficultLabels))


print("------------------- To  file---------------------------")
with open('features.txt', 'w') as f:
    for item in features:
        f.write("%s\n" % item)

with open('labels.txt', 'w') as f:
    for item in labels:
        f.write("%s\n" % item)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(features, labels)


test = english.featuresFromFile("./test/train_170.src.txt")
print(clf.predict(test))
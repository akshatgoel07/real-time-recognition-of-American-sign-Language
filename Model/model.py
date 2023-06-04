import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, KFold, cross_val_predict
from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from joblib import dump, load

## Define datset path
datasetPath = "Dataset/hand_dataset_1000_24.csv"

## Read dataset using pandas
dataset = pd.read_csv(datasetPath)

## Split dataset into X and y
## @param X: the landmark coodinates data
## @param y: the corresponding letter
X = dataset.drop('class', axis = 1)
y = dataset['class']

## Create normalizer and normalize coordinate data
normalizer = Normalizer().fit(X)

X = normalizer.transform(X)


## Code used to test various models accuracy score on data
cross_Validate = KFold(n_splits = 10, random_state=7, shuffle = True)

def test_model(modelName, model):
    print(f'Testing Model {modelName} ...')
    scoring = ['accuracy']
    scores = cross_validate(model, X, y, scoring=scoring, cv=cross_Validate, n_jobs=-1)

    y_pred = cross_val_predict(model, X, y, cv=cross_Validate)
    
    accuracy = np.mean(scores['test_accuracy'])
    print('Mean Accuracy: %.3f\n' % (accuracy))

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Calculate precision, recall, and F1-score
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    print('Specificity:', specificity)
    print('Sensitivity:', sensitivity)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    
    return accuracy

a = test_model("GNB", GaussianNB())

b = test_model("KNN", KNeighborsClassifier())

c = test_model("LR", LogisticRegression(max_iter = 1000))

d = test_model("DT", DecisionTreeClassifier())

e = test_model("SVC", SVC())

accuracy_scores = [a, b, c, d, e]
model_names = ["GNB", "KNN", "LR", "DT", "SVC"]
# Create the bar plot
plt.bar(model_names, accuracy_scores)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Different Models")

# Display the plot
plt.show()
## Classifier chosen in KNeighbors
classifier = KNeighborsClassifier()

## Train our classifier using dataset
classifier.fit(X, y)

## Dump classifier to be used in handDetection.py
dump(classifier, "model.joblib")
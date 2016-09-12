#!/usr/bin/python
'''
read me
http://broadwater.io/identifying-fraud-from-enron-email/
http://bl.ocks.org/dmenin/raw/d12a22521ad32cacc906/
https://discussions.udacity.com/t/how-to-start-the-final-project/177617/2
https://discussions.udacity.com/t/starting-on-the-project/30096/4

'''
import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import (dump_classifier_and_data, load_classifier_and_data,
test_classifier)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','total_stock_value','deferred_income','long_term_incentive'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.cross_validation import StratifiedShuffleSplit
folds = 1000
cv = StratifiedShuffleSplit(labels, folds, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []

for train_index, test_index in cv:
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    for ii in train_index:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_index:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    from sklearn import svm

    clf = tree.DecisionTreeClassifier() #svm.SVC()  #tree.DecisionTreeClassifier() #GaussianNB()
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    
    from sklearn.metrics import accuracy_score
    accuracy_scores.append( clf.score(features_test, labels_test) )
    
    from sklearn.metrics import precision_score, recall_score
    
    precision_scores.append( precision_score(labels_test, pred) )
    recall_scores.append( recall_score(labels_test, pred) )

accuracy = np.mean(accuracy_scores)
precision = np.mean(precision_scores)
recall = np.mean(recall_scores)

print "accuracy score: ", accuracy, " precision score: ", precision, " recall: ", recall

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

clf, dataset, feature_list = load_classifier_and_data()

test_classifier(clf, dataset, feature_list)

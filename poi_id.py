#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

r = 42

data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

data_dict.pop( "TOTAL", 0 ) # remove "TOTAL" line item
data_dict.pop( "THE TRAVEL AGENCY IN THE PARK", 0 ) # remove "THE TRAVEL AGENCY IN THE PARK" line item
data_dict.pop( "LOCKHART EUGENE E", 0 ) # remove "LOCKHART EUGENE E" line item

feature_list = ['poi',
               'bonus',
               'salary',
               'deferral_payments',
               'deferred_income',
               'director_fees',
               'exercised_stock_options',
               'expenses',
               'total_payments',
               'total_stock_value',
               'from_messages',
               'from_poi_to_this_person',
               'from_this_person_to_poi',
               'loan_advances',
               'long_term_incentive',
               'other',
               'restricted_stock',
               'restricted_stock_deferred',
               'salary',
               'shared_receipt_with_poi',
               'to_messages'
               ]

data = featureFormat(data_dict, feature_list)

import pprint
pp = pprint.PrettyPrinter(depth=6)

import copy
my_dataset = copy.deepcopy(data_dict)
my_feature_list = copy.deepcopy(feature_list)

for k in my_dataset.keys():
    my_dataset[k]['ratio_to_poi_to_all_sent']  = 0
    if (my_dataset[k]['from_poi_to_this_person'] != 'NaN') and (my_dataset[k]['from_messages'] != 'NaN') and (my_dataset[k]['from_messages'] != 0):
        my_dataset[k]['ratio_to_poi_to_all_sent'] = float(my_dataset[k]['from_this_person_to_poi'])/float(my_dataset[k]['from_messages'])

    my_dataset[k]['ratio_from_poi_to_all_received']  = 0
    if (my_dataset[k]['from_this_person_to_poi'] != 'NaN') and (my_dataset[k]['to_messages'] != 'NaN') and (my_dataset[k]['to_messages'] != 0):
        my_dataset[k]['ratio_from_poi_to_all_received'] = float(my_dataset[k]['from_poi_to_this_person'])/float(my_dataset[k]['to_messages'])


for i in ['ratio_to_poi_to_all_sent','ratio_from_poi_to_all_received']:
    if i not in my_feature_list:
        my_feature_list.append(i)




# ## Gaussian Naive Bayes

# from sklearn.grid_search import GridSearchCV
# from sklearn.naive_bayes import GaussianNB
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import Pipeline, FeatureUnion


# from sklearn.cross_validation import StratifiedShuffleSplit
# my_data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
# labels, feature_values = targetFeatureSplit(my_data)
# folds = 1000
# cv = StratifiedShuffleSplit(
#      labels, folds, random_state=r)

# clf = GaussianNB()
# steps = [
#     ('scale', MinMaxScaler()),
#     ('select_features',SelectKBest(f_classif)),
#     ('my_classifier', clf)
#     ]

# parameters = dict(select_features__k=[3,5,9,15,19,21,'all'])

# pipe = Pipeline(steps)

# grid = GridSearchCV(pipe, param_grid=parameters, cv=cv, verbose=1, scoring='f1', n_jobs=4)

# grid.fit(feature_values, labels)

# print("The best parameters are %s with a score of %0.4f"
#       % (grid.best_params_, grid.best_score_))

# # The best parameters are {'select_features__k': 'all'} with a score of 0.3248
# # Pipeline(steps=[('scale', MinMaxScaler(copy=True, feature_range=(0, 1))), ('select_features', SelectKBest(k='all', score_func=<function f_classif at 0x115589e60>)), ('my_classifier', GaussianNB())])
# # 	Accuracy: 0.53120	Precision: 0.19280	Recall: 0.78950	F1: 0.30991	F2: 0.48765
# # 	Total predictions: 15000	True positives: 1579	False positives: 6611	False negatives:  421	True negatives: 6389


# gnb_classifier = grid.best_estimator_

# # use test_classifier to evaluate
# test_classifier(gnb_classifier, my_dataset, my_feature_list)



# ## KNeighborsClassifier

# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.cross_validation import StratifiedShuffleSplit
# my_data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
# labels, feature_values = targetFeatureSplit(my_data)
# folds = 1000
# cv = StratifiedShuffleSplit(
#      labels, folds, random_state=r)

# clf = KNeighborsClassifier()
# steps = [
#     ('scale', MinMaxScaler()),
#     ('select_features',SelectKBest(f_classif)),
#     ('my_classifier', clf)
#     ]

# parameters = dict(select_features__k=[1,2,3,4,5,6,7,9,11,13,15,17,19,21], 
#               my_classifier__n_neighbors=[1,2,3,4,5,6,7,8,9,13,15,20])

# pipe = Pipeline(steps)

# grid = GridSearchCV(pipe, param_grid=parameters, cv=cv, verbose=1, scoring='f1', n_jobs=4)

# grid.fit(feature_values, labels)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

# # The best parameters are {'my_classifier__n_neighbors': 1, 'select_features__k': 6} with a score of 0.30


# knn_classifier = grid.best_estimator_

# # evaluate with test_classifier function
# test_classifier(knn_classifier, my_dataset, my_feature_list)

# # Pipeline(steps=[('scale', MinMaxScaler(copy=True, feature_range=(0, 1))), ('select_features', SelectKBest(k=6, score_func=<function f_classif at 0x115f53de8>)), ('my_classifier', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
# #            metric_params=None, n_jobs=1, n_neighbors=1, p=2,
# #            weights='uniform'))])
# # 	Accuracy: 0.80733	Precision: 0.30187	Recall: 0.33900	F1: 0.31936	F2: 0.33086
# # 	Total predictions: 15000	True positives:  678	False positives: 1568	False negatives: 1322	True negatives: 11432


## DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion


from sklearn.cross_validation import StratifiedShuffleSplit
my_data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, feature_values = targetFeatureSplit(my_data)
folds = 1000
cv = StratifiedShuffleSplit(
     labels, folds, random_state=r)

clf = DecisionTreeClassifier(random_state=r)
steps = [
    ('scale', MinMaxScaler()),
    ('select_features',SelectKBest(f_classif)),
    ('my_classifier', clf)
    ]

parameters = dict(select_features__k=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,'all'],
                  my_classifier__max_features=[None, 'auto', 'log2'],
                  my_classifier__criterion=['gini', 'entropy'],
                  my_classifier__min_samples_split=[2, 3, 4, 5, 10]
                 )

pipe = Pipeline(steps)

grid = GridSearchCV(pipe, param_grid=parameters, cv=cv, verbose=1, scoring='f1', n_jobs=4)

grid.fit(feature_values, labels)

print("The best parameters are %s with a score of %0.4f"
      % (grid.best_params_, grid.best_score_))

# The best parameters are {'my_classifier__min_samples_split': 10, 'select_features__k': 19, 'my_classifier__criterion': 'entropy', 'my_classifier__max_features': None} with a score of 0.3578

dt_classifier = grid.best_estimator_

test_classifier(dt_classifier, my_dataset, my_feature_list)


# Pipeline(steps=[('scale', MinMaxScaler(copy=True, feature_range=(0, 1))), ('select_features', SelectKBest(k=19, score_func=<function f_classif at 0x119eeb0c8>)), ('my_classifier', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
#             min_samples_split=10, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=42, splitter='best'))])
# 	Accuracy: 0.85120	Precision: 0.42961	Recall: 0.35400	F1: 0.38816	F2: 0.36692
# 	Total predictions: 15000	True positives:  708	False positives:  940	False negatives: 1292	True negatives: 12060

dump_classifier_and_data(clf, my_dataset, feature_list)

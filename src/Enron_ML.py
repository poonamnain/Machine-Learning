"""
Created on Fri Mar 9 11:31:20 2018

@author: poonam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle

import numpy as np
import math
import pandas as pd
import sklearn

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.pipeline import FeatureUnion

from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

cwd = os.getcwd()
print(cwd)

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

enron_data_df = pd.DataFrame.from_dict(data_dict, orient = 'index')
enron_data_df.shape

print(enron_data_df.head())
print(enron_data_df.describe())

print ("There are a total of {} people in the dataset." 
       .format(len(enron_data_df.index)))
print ("Out of which {} are POI and {} Non-POI." 
       .format(enron_data_df['poi'].value_counts()[True],
               enron_data_df['poi'].value_counts()[False]))
print ("Total number of email & financial features are {}, this includes POI identifier"
       .format(len(enron_data_df.columns)))


finance_cols = [ 'salary','deferral_payments', 'deferred_income','loan_advances', 'bonus','expenses', 
                'long_term_incentive','director_fees', 'other', 'total_payments']
for record in data_dict.values():
    for col in finance_cols:
        if record[col] == 'NaN':
           record[col] = 0
   
stock_cols = ['restricted_stock_deferred', 'exercised_stock_options', 
              'restricted_stock','total_stock_value']
for record in data_dict.values():
    for col in stock_cols:
        if record[col] == 'NaN':
           record[col] = 0

email_cols = [ 'to_messages', 'from_poi_to_this_person', 
              'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi']
# make all 0 except email_address
for record in data_dict.values():
    for col in email_cols:
        if record[col] == 'NaN':
           record[col] = 0


# Remove outlier TOTAL and THE TRAVEL AGENCY IN THE PARK from dict

data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)

# Remove email_address column

data_dict.pop('email_address', None)

print (len(data_dict))

# Note- features_list without email_address
features_list = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments',
       'loan_advances', 'bonus','restricted_stock_deferred',
       'deferred_income', 'total_stock_value', 'expenses',
       'from_poi_to_this_person', 'exercised_stock_options', 'from_messages',
       'other', 'from_this_person_to_poi', 'long_term_incentive',
       'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] 


#from feature_format import featureFormat, targetFeatureSplit (if separate file- else define functions here)

def featureFormat(dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    return_list = []
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print ("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append(float(value))

        append = True
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        if append:
            return_list.append( np.array(tmp_list) )
    return np.array(return_list)


def targetFeatureSplit( data ):
    target = []
    features = []
    for item in data:
        #print('target=' ,item[0])
        target.append( item[0] )
        features.append( item[1:] )
    return target, features


def get_k_best(data_dict, features_list, k):
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print("{0} best features: {1}\n".format(k, k_best_features.keys()))
    return k_best_features

k_best = get_k_best(data_dict, features_list, 10)
print (k_best)


# Add new features

for key in data_dict:
    if (data_dict[key]['total_payments'] != 0):
    	data_dict[key]["bonus_by_total"] = data_dict[key]['bonus']/data_dict[key]['total_payments']
    	data_dict[key]['salary_by_total'] = data_dict[key]['salary']/data_dict[key]['total_payments']
    else:
       data_dict[key]["bonus_by_total"] = 1
       data_dict[key]['salary_by_total'] = 1

    if (data_dict[key]['from_messages'] != 0):
    	data_dict[key]['to_poi_by_total_msg'] = data_dict[key]['from_this_person_to_poi']/data_dict[key]['from_messages']
    else:
        data_dict[key]['to_poi_by_total_msg'] = 1
    if (data_dict[key]['to_messages'] != 0):
    	data_dict[key]['from_poi_by_total_msg'] = data_dict[key]['from_poi_to_this_person']/data_dict[key]['to_messages']
    else:
        data_dict[key]['from_poi_by_total_msg'] = 1


predictors_best15 = ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments','loan_advances', 'bonus','restricted_stock_deferred',
       'deferred_income', 'total_stock_value', 'expenses','from_poi_to_this_person', 'exercised_stock_options', 'from_messages',
       'other', 'from_this_person_to_poi']


predictors_best12 = ['poi','exercised_stock_options', 'total_stock_value','bonus','salary','deferred_income','long_term_incentive',
 'restricted_stock', 'total_payments', 'shared_receipt_with_poi','loan_advances','expenses', 'from_poi_to_this_person']


predictors_best10 = ['poi','exercised_stock_options', 'total_stock_value','bonus','salary','deferred_income','long_term_incentive',
 'restricted_stock', 'total_payments', 'shared_receipt_with_poi','loan_advances']


predictors_best8 = ['poi','exercised_stock_options', 'total_stock_value','bonus','salary','deferred_income','long_term_incentive',
 'restricted_stock', 'total_payments']

predictors_10with_new = ['poi','exercised_stock_options', 'total_stock_value','bonus','salary','deferred_income','long_term_incentive', 
'bonus_by_total','salary_by_total', 'to_poi_by_total_msg', 'from_poi_by_total_msg']

predictors_8with_new = ['poi','exercised_stock_options', 'total_stock_value','bonus','salary', 
'bonus_by_total','salary_by_total', 'to_poi_by_total_msg', 'from_poi_by_total_msg']


data = featureFormat(data_dict, predictors_best15)
labels, features = targetFeatureSplit(data)

# Naive Beyes - GaussianNB
gnb_clf = GaussianNB()
scores = sklearn.cross_validation.cross_val_score(gnb_clf, features, labels)
print (scores)
print ('GaussianNB mean score:', scores.mean())


# Support Vector Machine (Classifier)
svc_clf = SVC()
scores = sklearn.cross_validation.cross_val_score(svc_clf, features, labels)
print (scores)
print ('SVC:', scores.mean())


# DecisionTree
dt_clf = DecisionTreeClassifier()
scores = sklearn.cross_validation.cross_val_score(dt_clf, features, labels)
print (scores)
print ('DecisionTree mean score:', scores.mean())


# AdaBoostClassifier
ab_clf = AdaBoostClassifier(n_estimators=100)
scores = sklearn.cross_validation.cross_val_score(ab_clf, features, labels)
print (scores) 
print ('AdaBoostClassifier mean score:', scores.mean())


# Logistic Regression
lreg_clf = LogisticRegression()
scores = sklearn.cross_validation.cross_val_score(lreg_clf, features, labels)
print (scores) 
print ('LogisticRegression mean score:', scores.mean())



# split into test and training data
X_training_features, X_test_features, y_train_poi, y_test_poi = train_test_split(features, labels, test_size=0.33, random_state=42, stratify= labels)

print ('Data has been split---------imp step-----')
#print(X_training_features)
#print(X_test_features)

print ('feature selection and parameter tuning begins to identify the best classifiers')
# precision is (true positives)/ (true positives + false positives)
# recall is (true positives)/ (true positives + false negatives)

k= 15
min_precision = 0.3
min_recall = 0.3

# make classifiers Use GridSearchCV to automate the process of finding the optimal features and parameters on different algorithms

print ('################### Try SVM ###################################')
# SVM - 1
svm_clf_1 = SVC(random_state=0)

pipe = Pipeline([
	('scaler',StandardScaler()),
	("kbest", SelectKBest(k=k)),
	('classification', svm_clf_1)
])

# Check the parameters that can be set for DecisionTree Classifier, and create a param_grid 
estimated = svm_clf_1.get_params().keys()
print ('param_keys########################',estimated)
#dict_keys(['C', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'])

param_grid = ([{'classification__C': [100],
	'classification__gamma': [0.1],
	'classification__degree':[2],
	'classification__kernel': ['poly'],
	'classification__max_iter':[5]
}])
svm_clf_1 = GridSearchCV(pipe, param_grid, scoring='recall')
svm_clf_1.fit(X_training_features, y_train_poi).best_estimator_

clf_best = svm_clf_1.best_estimator_

y_poi_predicted = clf_best.predict(X_test_features)
f1_tree = f1_score(y_test_poi, y_poi_predicted)
precision_tree = precision_score(y_test_poi, y_poi_predicted)
recall_tree = recall_score(y_test_poi, y_poi_predicted)
print ('f1 score for DecisionTreeClassifier', f1_tree)
print ('precision for DecisionTreeClassifier', precision_tree)
print ('recall for DecisionTreeClassifier', recall_tree)

if (precision_tree >= min_precision) & (recall_tree >= min_recall):
	print ('DecisionTree is a good classifier with set parameters')
else:
	print ('Low precision and recall, DecisionTree is not a good classifier with set parameters')



print ('################### Try DecisionTreeClassifier ###################################')
# DecisionTree - 1
tree_clf_1 = tree.DecisionTreeClassifier(random_state=0)

# create feature union
features_pipeline = []
features_pipeline.append(('pca', PCA(n_components=3)))
features_pipeline.append(('select_best', SelectKBest(k=k)))
feature_union = FeatureUnion(features_pipeline)


# Create a pipeline with feature selection and classification
pipe = Pipeline([
  ('feature_union', feature_union),
  ('feature_selection', SelectKBest(k=k)),
  ('classification', tree_clf_1)
])

# Check the parameters that can be set for DecisionTree Classifier, and create a param_grid 
estimated = tree_clf_1.get_params().keys()
print ('param_keys########################',estimated)

param_grid = {'classification__class_weight': [None],
'classification__criterion': ["gini", "entropy"],
'classification__min_samples_split': [2, 10, 20],
'classification__max_depth': [None, 2, 5, 10],
'classification__min_samples_leaf': [1, 5, 10],
'classification__max_leaf_nodes': [None, 5, 10, 20],
'classification__splitter': ["random"]
}

scorer = make_scorer(f1_score)
tree_clf_1 = GridSearchCV(pipe, param_grid = param_grid, scoring= scorer, cv = 5)
tree_clf_1.fit(X_training_features, y_train_poi)

clf_best = tree_clf_1.best_estimator_

y_poi_predicted = clf_best.predict(X_test_features)
f1_tree = f1_score(y_test_poi, y_poi_predicted)
precision_tree = precision_score(y_test_poi, y_poi_predicted)
recall_tree = recall_score(y_test_poi, y_poi_predicted)
print ('f1 score for DecisionTreeClassifier', f1_tree)
print ('precision for DecisionTreeClassifier', precision_tree)
print ('recall for DecisionTreeClassifier', recall_tree)

if (precision_tree >= min_precision) & (recall_tree >= min_recall):
	print ('DecisionTree is a good classifier with set parameters')
else:
	print ('Low precision and recall, DecisionTree is not a good classifier with set parameters')


print ('################### Try AdaBoostClassifier ###################################')
# AdaBoostClassifier - 2
ab_clf_1 = AdaBoostClassifier()

pipe = Pipeline([
  ('feature_selection', SelectKBest(k=k)),
  ('classification', ab_clf_1)
])

# Check the parameters that can be set for AdaBoostClassifier, and create a param_grid 
estimated = ab_clf_1.get_params().keys()
print ('param_keys########################',estimated)

param_grid = {'classification__n_estimators': [10, 50, 100]}

scorer = make_scorer(f1_score)
ab_clf_1 = GridSearchCV(pipe, param_grid = param_grid, scoring= scorer)
ab_clf_1.fit(X_training_features, y_train_poi)

scores = sklearn.cross_validation.cross_val_score(ab_clf_1, features, labels)
print (scores) 
print ('AdaBoostClassifier mean score:', scores.mean())

clf_best = ab_clf_1.best_estimator_

y_poi_predicted = clf_best.predict(X_test_features)
f1_ada = f1_score(y_test_poi, y_poi_predicted)
precision_ada = precision_score(y_test_poi, y_poi_predicted)
recall_ada= recall_score(y_test_poi, y_poi_predicted)
print ('f1 score for AdaBoostClassifier', f1_ada)
print ('precision for AdaBoostClassifier', precision_ada)
print ('recall for AdaBoostClassifier', recall_ada)

if (precision_ada >= min_precision) & (recall_ada >= min_recall):
	print ('AdaBoostClassifier is a good classifier with set parameters')
else:
	print ('Low precision and recall, AdaBoostClassifier is not a good classifier with set parameters')


print ('################### Try RandomForestClassifier ###################################')

rf_clf_1 = RandomForestClassifier()

# create feature union
features_pipeline = []
features_pipeline.append(('pca', PCA(n_components=4)))
features_pipeline.append(('select_best', SelectKBest(k=k)))
feature_union = FeatureUnion(features_pipeline)

# find best fitting parameter
pipe = Pipeline([
  	('feature_union', feature_union),
	('classification', rf_clf_1)
])

param_grid = {'classification__n_estimators': [100,250,500],
               'classification__min_samples_split' :[2,3,4,5],
               'classification__min_samples_leaf' : [1,2,3]
             }

# Check the parameters that can be set for RandomForestClassifier, and create a param_grid 
estimated = rf_clf_1.get_params().keys()
print ('param_keys########################',estimated)

scorer = make_scorer(f1_score)
rf_clf_1 = GridSearchCV(pipe, param_grid = param_grid, scoring= scorer)
rf_clf_1.fit(X_training_features, y_train_poi)

scores = sklearn.cross_validation.cross_val_score(rf_clf_1, features, labels, cv=2)
print (scores) 
print ('RandomForestClassifier mean score:', scores.mean())

clf_best = rf_clf_1.best_estimator_

y_poi_predicted = clf_best.predict(X_test_features)
f1_rf = f1_score(y_test_poi, y_poi_predicted)
precision_rf = precision_score(y_test_poi, y_poi_predicted)
recall_rf  = recall_score(y_test_poi, y_poi_predicted)
print ('f1 score for RandomForestClassifier', f1_rf)
print ('precision for RandomForestClassifier', precision_rf)
print ('recall for RandomForestClassifier', recall_rf)

if (precision_rf >= min_precision) & (recall_rf >= min_recall):
	print ('RandomForestClassifier is a good classifier with set parameters')
else:
	print ('Low precision and recall, RandomForestClassifier is not a good classifier with set parameters')

print ('Out of the 3 classifiers the one with max recall is most suitable for Enron Data')
print ('DecisionTreeClassifier is with max recall and so is most suitable for Enron Data')


print('######Generating the necessary .pkl files for validating your result#####################')
#One of this is best svm_clf_1, tree_clf_1, ab_clf_1, rf_clf_1
clf = tree_clf_1.best_estimator_
my_dataset = data_dict
features_list = predictors_best15

pickle.dump(clf, open("my_classifier.pkl", "wb"))
pickle.dump(my_dataset, open("my_dataset.pkl", "wb"))
pickle.dump(features_list, open("my_feature_list.pkl", "wb"))
################################ EOF #######################

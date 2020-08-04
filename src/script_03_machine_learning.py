#! /usr/bin/env python

# script_03_machine_learning.py
# Olivia Lin, Mani Kohli Nov 23 2018
#
# This script takes in the cleaned version of dataset and returns the summary tables of machile learning for our project.
# This script should be run under the root folder.

# Usage: python src/script_03_machine_learning.py
# Example: python src/script_03_machine_learning.py data/tidy_data_ml.csv results/depth_summary.csv results/feature_summary.csv


# import libraries
import argparse
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize, scale, Normalizer, StandardScaler
import matplotlib.pyplot as plt
#import graphviz
#from sklearn.tree import export_graphviz


# parse/define command line arguments here
parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("output_file_depth")
parser.add_argument("output_file_feature")
args = parser.parse_args()

# loading data
data = pd.read_csv(args.input_file)

# split data into trainning set and testing set
X = data.iloc[:, 0:9]
y = data.iloc[:, 9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# standardizing the trainning and testing set
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# find the optimal depth
accuracy = []
for i in range(1, 21):
    model = tree.DecisionTreeClassifier(max_depth = i, random_state = 222)
    score = cross_val_score(model, X_train, y_train, cv = 5).mean()
    accuracy.append(score)
depth = accuracy.index(max(accuracy))+1

# use decision tree with optimal max_depth parameter
model = tree.DecisionTreeClassifier(max_depth = depth, random_state = 222)
model.fit(X_train, y_train)

# creating summary table for max depth
depth_range = [i for i in range(1, 21)]
depth_summary = pd.DataFrame([depth_range, accuracy], index = ["max_depth", "accuracy"]).T
# store depth summary table to csv file
depth_summary.to_csv(args.output_file_depth)

# creating feature summary table
feature = list(X.columns)
importance = list(model.feature_importances_)
feature_summary = pd.DataFrame([feature, importance], index = ["variable", "feature_importance"]).T
# store feature summary table to csv file
feature_summary.to_csv(args.output_file_feature)


# The belowing code is for creating decision tree plot
# select features based feauture importance (for the purpose of readable decision tree graph)

#X_selected = X.ix[:,['married', 'capitalGain', 'education']]

# spliting the data based after selected features

#X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state = 42)

# standardizing agian for the new data

#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# fit the model again (using depth 6 for readable decision tree graph)

#model = tree.DecisionTreeClassifier(max_depth = 6, random_state = 222)
#model.fit(X_train, y_train)

# function for sace the tree (from 571 lecture)

#def save_decision_tree(model,save_file_prefix = args.output_file_tree, **kwargs):
#    """
#    Save the decision tree model as a png
#    """
#    feature_cols = ['married', 'capitalGain', 'education']
#    class_names = ['low income','high income']
#    dot_data = export_graphviz(model, out_file = None,
#                             feature_names = feature_cols,
#                             class_names = class_names,
#                             filled = True, rounded = True,
#                             special_characters = True, **kwargs)

#    graph = graphviz.Source(dot_data, format = "png")
#    graph.render(save_file_prefix)

# call the function

#save_decision_tree(model)

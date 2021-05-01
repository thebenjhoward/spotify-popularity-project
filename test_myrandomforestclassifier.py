##############################################
# Programmer: Ben Howard and Elizabeth Larson
# Class: CPSC 322-01, Spring 2021
# Final project
# 05/05/21
#
# Sources:
#   TODO: Finish this
# 
# Description: This program tests MyRandomForestClassifier fit() and predict().
##############################################

# TODO: Finish all TODOs

import numpy as np # TODO: Needed?

from mysklearn.myrandomforestclassifier import MyRandomForestClassifier

# For dict dataset (convert to a MyPyTable)
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 

header = ["level", "lang", "tweets", "phd", "interviewed_well"]
table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]
# TODO: Finish all answers
forest_class1_fit_answer = []
forest_class2_fit_answer = []
forest_class3_fit_answer = []
forest_class1_predict_answer = []
forest_class2_predict_answer = []
forest_class3_predict_answer = []

def test_random_forest_classifier_fit():
    forest_class1 = MyRandomForestClassifier(4, 2, 2)
    X_train = table
    y_train = []
    for row in X_train:
        y_train.append(row[-1])
    forest_class1.fit(X_train, y_train)
    # assert forest_class1.forest == forest_class1_fit_answer

    # TODO: Fit a few more times with different N, M, and F values
    # N = 20, M = 7, and F = 2
    # ... and then another

    assert False == True # TODO: Finish this

"""def test_random_forest_classifier_predict():
    assert False == True # TODO: Finish this"""
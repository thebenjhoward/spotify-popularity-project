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

# TODO: Interview dataset with different parameters (such as N = 20, M = 7, and F = 2)
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

def test_random_forest_classifier_fit():
    assert False == True # TODO: Finish this

def test_random_forest_classifier_predict():
    assert False == True # TODO: Finish this
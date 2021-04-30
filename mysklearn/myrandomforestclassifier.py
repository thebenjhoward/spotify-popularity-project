##############################################
# Programmer: Ben Howard and Elizabeth Larson
# Class: CPSC 322-01, Spring 2021
# Final project
# 05/05/21
#
# Sources:
#   TODO: Finish this
# 
# Description: This program handles a random forest classifier, utalizing the
#              the basic random forest classifier algorithm. TODO: Finish this
##############################################

# TODO: Finish all TODOs

import mysklearn.myutils as myutils

class MyRandomForestClassifier:
    def __init__(self):
        """TODO: Finish header"""
        pass # TODO: Finish

    def fit(self):
        """TODO: Finish header"""
        pass # TODO: Finish

    def predict(self):
        """TODO: Finish header"""
        pass # TODO: Finish

"""
1. Generate a random stratified test set consisting of one third of the original data set, with the remaining two thirds of the instances forming the "remainder set".
2. Generate N "random" decision trees using bootstrapping (giving a training and validation set) over the remainder set. At each node, build your decision trees by randomly selecting F of the remaining attributes as candidates to partition on. This is the standard random forest approach discussed in class. Note that to build your decision trees you should still use entropy; however, you are selecting from only a (randomly chosen) subset of the available attributes.
3. Select the M most accurate of the N decision trees using the corresponding validation sets.
4. Use simple majority voting to predict classes using the M decision trees over the test set.

Note that N, M, and F are all parameters that need to be tuned.
Run your random forest algorithm for each data set to see the variation of results for different values of the parameters  N, M, and F.
Note that for each setting of  N, M, and F, you will need to run your program multiple times because of the randomly generated remainder set to get a sense of the settings (e.g., you might run each setting 5 times).
You should try a wide range of values including large values for  N.
Report the results (i.e., the values for  N, M, and F , the accuracy, and the confusion matrices) that seem to give the best results for your dataset.
You should output the accuracy of the corresponding single "normal" decision tree as well for comparison. Report on the settings you tried and the results you obtained.
"""
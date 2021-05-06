##############################################
# Programmer: Ben Howard and Elizabeth Larson
# Class: CPSC 322-01, Spring 2021
# Final project
# 05/05/21
#
# No sources to cite.
# 
# Description: This program handles a random forest classifier, utalizing the
#              the basic random forest classifier algorithm to fit and predict.
##############################################


import random # For majority voting
random.seed(0)
import mysklearn.myutils as myutils
from mysklearn.myclassifiers import MyDecisionTreeClassifier # For tree building

class MyRandomForestClassifier:
    def __init__(self, N, M, F):
        """Initializer for MyRandomForestClassifier.

        Args:
            N(int): How many trees we're initially making
            M(int): How many trees we're pruning down to
            F(int): Attributes to choose from when building trees
        """

        self.N = N
        self.M = M
        self.F = F
        self.X_train = None
        self.y_train = None
        self.forest = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the random forest algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Random forest algorithm (from project assignment description):
                1. Generate a random stratified test set consisting of one third of the original data set, with the remaining two thirds of the instances forming the "remainder set".
                2. Generate N "random" decision trees using bootstrapping (giving a training and validation set) over the remainder set.
                    At each node, build your decision trees by randomly selecting F of the remaining attributes as candidates to partition on.
                    This is the standard random forest approach discussed in class.
                    Note that to build your decision trees you should still use entropy;
                    however, you are selecting from only a (randomly chosen) subset of the available attributes.
                3. Select the M most accurate of the N decision trees using the corresponding validation sets.
                4. Use simple majority voting to predict classes using the M decision trees over the test set.
                    See predict() for this step
        """

        # Set X_train and y_train
        self.X_train = X_train
        self.y_train = y_train

        # Generate test and remainder sets
        _, remainder_set = myutils.make_test_and_remainder_sets(X_train)

        # Create N "random" decision trees using bootstrapping over the remainder set
        N_trees = []
        for _ in range(self.N):
            boostrap_sample = myutils.compute_bootstrap_sample(remainder_set) # Grab a bootstrap sample

            # Build a tree
            tree_class = MyDecisionTreeClassifier()
            tree_class.X_train = X_train
            tree_class.y_train = y_train
            tree_class.fit(boostrap_sample, tree_class.y_train, self.F)

            N_trees.append(tree_class.tree) # Keep track of the tree

        # Find the index of the attribute we're predicting
        # e.g. col #0 in the interview dataset is ['Senior', 'Senior', 'Mid', 'Junior', 'Junior', 'Junior', 'Mid', 'Senior', 'Senior', 'Junior', 'Senior', 'Mid', 'Mid', 'Junior']
        col_values = []
        for col_index in range(len(X_train[0])):
            new_col = []
            for row in X_train:
                new_col.append(row[col_index])
            col_values.append(new_col)
        predict_col_index = col_values.index(self.y_train)

        # Pick the M most accurate trees from N_trees to populate the forest
        self.forest = myutils.find_most_accurate_trees(N_trees, self.M, predict_col_index, self.X_train, MyDecisionTreeClassifier)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)

        Notes:
            Random forest algorithm (from project assignment description):
                4. Use simple majority voting to predict classes using the M decision trees over the test set.
        """

        y_predicted = []

        # Grab all possible predict values (e.g. "False" and "True" for interview dataset)
        possible_predict_values = []
        for value in self.y_train:
            if not value in possible_predict_values:
                possible_predict_values.append(value)

        # Make X_test predictions on all trees
        all_trees_predictions = []
        for tree in self.forest:
            for X_test_value in X_test:
                prediction = myutils.predict_recursive_helper(tree, X_test_value)
                if prediction == None: # Pick a random row and use that as the prediction
                    random_index = random.randint(0, (len(self.y_train) - 1))
                    prediction = self.y_train[random_index]
                all_trees_predictions.append(prediction)

        # Perform majority voting to find the final/true prediction
        predictions = []
        prediction_counts = []
        for prediction in all_trees_predictions:
            if not prediction in prediction_counts:
                if prediction in possible_predict_values: # For predicting "att0" bug
                    predictions.append(prediction)
                    prediction_counts.append(all_trees_predictions.count(prediction))

        # If we have a short enough set of predictions (aka they all predict 1 value), take note of that value
        largest_count = max(prediction_counts)
        index_of_largest_count = prediction_counts.index(largest_count)
        y_predicted.append(predictions[index_of_largest_count])

        return y_predicted
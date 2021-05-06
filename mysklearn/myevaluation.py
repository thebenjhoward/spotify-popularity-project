import mysklearn.myutils as myutils
import random
from math import ceil
import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        random.seed(random_state)
    
    if shuffle:
        Xy = list(zip(X, y))
        random.shuffle(Xy)
        X, y = map(list, zip(*Xy))
    
    if(type(test_size) == int):
        split_point = len(X) - test_size
    else:
        split_point = len(X) - int(ceil(test_size * len(X)))

    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]

    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    train, tests = [], []

    large_count = len(X) % n_splits
    width = len(X) // n_splits + 1
    offset = 0

    indices = list(range(len(X)))

    for _i in range(large_count):
        tests.append(indices[offset:offset + width])
        offset+= width
    
    width -= 1
    for _i in range (n_splits - large_count):
        tests.append(indices[offset:offset + width])
        offset+= width


    for t1 in tests:
        train_set = []
        for t2 in tests:
            if(t2[0] != t1[0]): # we know there is no overlap and this saves massive time
                train_set.extend(t2)
        train.append(train_set)

    return train, tests

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        There is a lot of bloat because I tried to make it work with more than 2 categories.
        I didn't test this but theoretically it should
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    bins = myutils.separate_by_value(y)
    distros = {}
    factor_tracker = [] ## keeps track of which values are evenly divided by n_samples
                        ## important for ensuring fold size is propper

    for val, indices in bins.items():
        distros[val] = []
        num_big = len(indices) % n_splits
        factor_tracker.append(num_big == 0)
        
        width = len(indices) // n_splits + 1

        for i in range(n_splits):
            if(i == num_big):
                width -= 1
            distros[val].append(width)
    
    reverse = False
    for i, val in enumerate(distros):
        if(not factor_tracker[i]):
            if(reverse):
                distros[val].reverse()
            reverse = not reverse

    trains, tests = [], []
    indices = {x: 0 for x in bins}

    for i in range(n_splits):
        test = []
        for val in bins:
            vcount = distros[val][i]
            test.extend(bins[val][indices[val]:indices[val] + vcount])
            indices[val] += vcount
        tests.append(test)

    for test in tests:
        train_set = []
        for t2 in tests:
            if(test[0] != t2[0]): # saves massive time
                train_set.extend(t2)
        trains.append(train_set)
    
    return trains, tests

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    
    for true, pred in zip(y_true, y_pred):
        matrix[labels.index(true)][labels.index(pred)] += 1

    return matrix
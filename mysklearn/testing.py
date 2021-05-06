from itertools import combinations
from mysklearn.myevaluation import stratified_kfold_cross_validation, confusion_matrix
#from mysklearn.myevaluation import kfold_cross_validation, confusion_matrix
from altsklearn.myclassifiers import MyDecisionTreeClassifier
import mysklearn.myutils as myutils

def elem_subsets(attribs, size_range):
    """ Generates a set of subsets of elements to use for attribute selection
    
    Args:
        attribs(list): attributes to select from
        size_range(iterable): range of testing set sizes to generate attributes from
    """

    subsets = []
    for i in size_range:
        subsets.extend(combinations(attribs, i))
    
    return subsets

def bin_data(data, bins, labels, index=None):
    for j in range(len(data)):
        binned = False
        for i, bn in enumerate(bins):
            if(index is not None):
                if((data[j][index] >= bn[0] and data[j][index] < bn[1]) or (bn[0] == bn[1] and data[j][index] == bn[0])):
                    data[j][index] = labels[i]
                    binned = True
                    break
            else:
                if((data[j] >= bn[0] and data[j] < bn[1]) or (bn[0] == bn[1] and data[j] == bn[0])):
                    data[j] = labels[i]
                    binned = True
                    break
 
        if(not binned and index is None):
            raise Exception("Item out of range of bins: {}".format(data[j]))
        elif(not binned):
            raise Exception("Item out of range of bins: {}".format(data[j][index]))
            

def chunk_subsets(subsets, chunk_size):
    for i in range(0, len(subsets), chunk_size):
        yield subsets[i : i + chunk_size]


def validate_classifier(X, y, labels, attributes, classifier):
    """ Runs k-fold cross validatation with k = 10 on the classifier type given

    Args:
        X: base instance set for the dataset
        y: base class set for the dataset
        labels: range of y
        attributes: attribute indices to use for validation
        classifier(class instance): initialized classifier class that will be reused
        reuse(bool): for naive bayes classifiers, decide whether to reuse
    
    Returns:
        accuracy(float): the accuracy of the classifier overall
        confusion_matrix: Confusion matrix to represent classifications overall
    """

    train_folds, test_folds = stratified_kfold_cross_validation(X, y, n_splits=5)
    # train_folds, test_folds = kfold_cross_validation(X, n_splits=5)
    
    preds, trues = [], []
    correct = 0

    for train_fold, test_fold in zip(train_folds, test_folds):
        X_train, y_train, X_test, y_test = [], [], [], []
        for i in train_fold:
            X_train.append([X[i][j] for j in attributes])
            y_train.append(y[i])
        
        for i in test_fold:
            X_test.append([X[i][j] for j in attributes])
            y_test.append(y[i])
        
        if(type(classifier) == MyDecisionTreeClassifier):
            classifier.fit(X_train, y_train)
        else:
            classifier.fit(X_train, y_train)

        pred = classifier.predict(X_test)
        preds.extend(pred)
        trues.extend(y_test)
    
    for pred, true in zip(preds, trues):
        if(pred == true):
            correct += 1

    accuracy = correct / len(trues)

    conf_matrix = confusion_matrix(trues, preds, labels)

    return accuracy, conf_matrix

def gen_validation_result(attribs, params, labels, accuracy, matrix):
    if(params != ""):
        result =("attribs: {}\n"
                 "params: {}\n"
                 "accuracy: {}\n"
                 "confusion matrix:\n"
                 "{}\n").format(str(attribs), params, accuracy, myutils.format_confusion_matrix(matrix, ["Popularity", *labels]))
    else:
        result = ("attribs: {}\n"
                 "accuracy: {}\n"
                 "confusion matrix:\n"
                 "{}\n").format(str(attribs), accuracy, myutils.format_confusion_matrix(matrix, ["Popularity", *labels]))

    return result
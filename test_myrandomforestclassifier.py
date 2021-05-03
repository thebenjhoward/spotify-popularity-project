##############################################
# Programmer: Ben Howard and Elizabeth Larson
# Class: CPSC 322-01, Spring 2021
# Final project
# 05/05/21
#
# No sources to cite.
# 
# Description: This program tests MyRandomForestClassifier fit() and predict().
##############################################


from mysklearn.myrandomforestclassifier import MyRandomForestClassifier

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
forest_class1_fit_answer = [['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att1', ['Value', 'Python', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 2, 2]], ['Value', 'True', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 2, 2]], ['Value', 'yes', ['Leaf', 'False', 2, 2]]]]]], ['Value', 'R', ['Leaf', 'True', 1, 3]]]], ['Value', 'Mid', ['Leaf', 'True', 5, 5]], ['Value', 'Senior', ['Leaf', 'True', 2, 5]]]], ['Value', 'yes', ['Attribute', 'att4', ['Value', 'False', ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'True', 2, 2]], ['Value', 'R', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 2, 2]], ['Value', 'yes', ['Leaf', 'False', 2, 2]]]]]], ['Value', 'Mid', ['Leaf', 'False', 2, 2]], ['Value', 'Senior', ['Leaf', 'False', 2, 2]]]], ['Value', 'True', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 3, 3]], ['Value', 'Mid', ['Leaf', 'False', 1, 3]], ['Value', 'Senior', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 2, 2]], ['Value', 'yes', ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'True', 2, 2]], ['Value', 'R', ['Leaf', 'False', 2, 2]]]]]]]]]]], ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'False', 1, 10]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'True', 2, 5]], ['Value', 'R', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 3, 3]], ['Value', 'yes', ['Leaf', 'True', 3, 3]]]]]], ['Value', 'Mid', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'True', 2, 2]], ['Value', 'R', ['Leaf', 'True', 2, 2]]]], ['Value', 'yes', ['Leaf', 'False', 2, 2]]]], ['Value', 'Senior', ['Leaf', 'True', 7, 7]]]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 2, 2]], ['Value', 'Mid', ['Leaf', 'True', 1, 2]], ['Value', 'Senior', ['Leaf', 'False', 1, 2]]]]]]]]
forest_class2_fit_answer = [['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 1, 5]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 4, 4]], ['Value', 'Python', ['Leaf', 'True', 1, 4]], ['Value', 'R', ['Leaf', 'False', 3, 4]]]], ['Value', 'yes', ['Leaf', 'False', 4, 4]]]]]], ['Value', 'Mid', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 2, 2]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'False', 1, 2]], ['Value', 'yes', ['Leaf', 'True', 1, 2]]]]]], ['Value', 'Senior', ['Leaf', 'True', 2, 9]]], ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 1, 5]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 4, 4]], ['Value', 'Python', ['Leaf', 'True', 1, 4]], ['Value', 'R', ['Leaf', 'False', 3, 4]]]], ['Value', 'yes', ['Leaf', 'False', 4, 4]]]]]], ['Value', 'Mid', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 2, 2]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'False', 1, 2]], ['Value', 'yes', ['Leaf', 'True', 1, 2]]]]]], ['Value', 'Senior', ['Leaf', 'True', 2, 9]]], ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 1, 5]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 4, 4]], ['Value', 'Python', ['Leaf', 'True', 1, 4]], ['Value', 'R', ['Leaf', 'False', 3, 4]]]], ['Value', 'yes', ['Leaf', 'False', 4, 4]]]]]], ['Value', 'Mid', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 2, 2]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'False', 1, 2]], ['Value', 'yes', ['Leaf', 'True', 1, 2]]]]]], ['Value', 'Senior', ['Leaf', 'True', 2, 9]]], ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 1, 5]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 4, 4]], ['Value', 'Python', ['Leaf', 'True', 1, 4]], ['Value', 'R', ['Leaf', 'False', 3, 4]]]], ['Value', 'yes', ['Leaf', 'False', 4, 4]]]]]], ['Value', 'Mid', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 2, 2]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'False', 1, 2]], ['Value', 'yes', ['Leaf', 'True', 1, 2]]]]]], ['Value', 'Senior', ['Leaf', 'True', 2, 9]]], ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 1, 5]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'False', 4, 4]], ['Value', 'Python', ['Leaf', 'True', 1, 4]], ['Value', 'R', ['Leaf', 'False', 3, 4]]]], ['Value', 'yes', ['Leaf', 'False', 4, 4]]]]]], ['Value', 'Mid', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 2, 2]], ['Value', 'True', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'False', 1, 2]], ['Value', 'yes', ['Leaf', 'True', 1, 2]]]]]], ['Value', 'Senior', ['Leaf', 'True', 2, 9]]], ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Python', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 3, 3]], ['Value', 'Mid', ['Leaf', 'True', 1, 3]], ['Value', 'Senior', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 2, 2]], ['Value', 'True', ['Leaf', 'False', 2, 2]]]]]], ['Value', 'R', ['Leaf', 'True', 3, 3]]]], ['Value', 'yes', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'False', 4, 4]], ['Value', 'True', ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'False', 1, 4]], ['Value', 'R', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 1, 3]], ['Value', 'Mid', ['Leaf', 'True', 3, 3]], ['Value', 'Senior', ['Leaf', 'True', 2, 3]]]]]]]]]], ['Value', 'yes', ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'False', 2, 2]], ['Value', 'R', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'False', 2, 2]], ['Value', 'True', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 2, 2]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 2, 2]], ['Value', 'Mid', ['Leaf', 'True', 2, 2]], ['Value', 'Senior', ['Leaf', 'True', 2, 2]]]]]]]]]]], ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Attribute', 'att1', ['Value', 'Python', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 3, 3]], ['Value', 'Mid', ['Leaf', 'True', 1, 3]], ['Value', 'Senior', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 2, 2]], ['Value', 'True', ['Leaf', 'False', 2, 2]]]]]], ['Value', 'R', ['Leaf', 'True', 3, 3]]]], ['Value', 'yes', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'False', 4, 4]], ['Value', 'True', ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'False', 1, 4]], ['Value', 'R', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 1, 3]], ['Value', 'Mid', ['Leaf', 'True', 3, 3]], ['Value', 'Senior', ['Leaf', 'True', 2, 3]]]]]]]]]], ['Value', 'yes', ['Attribute', 'att1', ['Value', 'Python', ['Leaf', 'False', 2, 2]], ['Value', 'R', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'False', 2, 2]], ['Value', 'True', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 2, 2]], ['Value', 'yes', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'True', 2, 2]], ['Value', 'Mid', ['Leaf', 'True', 2, 2]], ['Value', 'Senior', ['Leaf', 'True', 2, 2]]]]]]]]]]]]
forest_class3_fit_answer = [['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'True', 1, 10]], ['Value', 'Python', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 1, 4]], ['Value', 'True', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 1, 3]], ['Value', 'Mid', ['Leaf', 'True', 2, 3]], ['Value', 'Senior', ['Leaf', 'True', 3, 3]]]]]], ['Value', 'R', ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 3, 3]], ['Value', 'yes', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 3, 3]], ['Value', 'True', ['Leaf', 'True', 3, 3]]]], ['Value', 'yes', ['Leaf', 'True', 3, 3]]]]]], ['Value', 'Mid', ['Leaf', 'False', 5, 5]], ['Value', 'Senior', ['Leaf', 'False', 2, 5]]]]], ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'True', 1, 10]], ['Value', 'Python', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 1, 4]], ['Value', 'True', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 1, 3]], ['Value', 'Mid', ['Leaf', 'True', 2, 3]], ['Value', 'Senior', ['Leaf', 'True', 3, 3]]]]]], ['Value', 'R', ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 3, 3]], ['Value', 'yes', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 3, 3]], ['Value', 'True', ['Leaf', 'True', 3, 3]]]], ['Value', 'yes', ['Leaf', 'True', 3, 3]]]]]], ['Value', 'Mid', ['Leaf', 'False', 5, 5]], ['Value', 'Senior', ['Leaf', 'False', 2, 5]]]]], ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'True', 1, 10]], ['Value', 'Python', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 1, 4]], ['Value', 'True', ['Attribute', 'att0', ['Value', 'Junior', ['Leaf', 'False', 1, 3]], ['Value', 'Mid', ['Leaf', 'True', 2, 3]], ['Value', 'Senior', ['Leaf', 'True', 3, 3]]]]]], ['Value', 'R', ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 3, 3]], ['Value', 'yes', ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att4', ['Value', 'False', ['Leaf', 'True', 3, 3]], ['Value', 'True', ['Leaf', 'True', 3, 3]]]], ['Value', 'yes', ['Leaf', 'True', 3, 3]]]]]], ['Value', 'Mid', ['Leaf', 'False', 5, 5]], ['Value', 'Senior', ['Leaf', 'False', 2, 5]]]]]]
forest_class1_predict_answer = ['True']
forest_class2_predict_answer = ['True']
forest_class3_predict_answer = ['True']

def test_random_forest_classifier_fit():
    # Load X_train and y_train for all
    X_train = table
    y_train = []
    for row in X_train:
        y_train.append(row[-1])

    # Test 1: N = 4, M = 2, F = 2
    forest_class1 = MyRandomForestClassifier(4, 2, 2)
    forest_class1.fit(X_train, y_train)
    assert forest_class1.forest == forest_class1_fit_answer

    # Test 2: N = 10, M = 7, and F = 2
    forest_class2 = MyRandomForestClassifier(10, 7, 2)
    forest_class2.fit(X_train, y_train)
    assert forest_class2.forest == forest_class2_fit_answer
    
    # Test 3: N = 10, M = 3, and F = 1
    forest_class3 = MyRandomForestClassifier(10, 3, 2)
    forest_class3.fit(X_train, y_train)
    assert forest_class2.forest == forest_class2_fit_answer

def test_random_forest_classifier_predict():
    # Load X_train and y_train for all
    X_train = table
    y_train = []
    for row in X_train:
        y_train.append(row[-1])

    # Test 1: N = 4, M = 2, F = 2
    forest_class1 = MyRandomForestClassifier(4, 2, 2)
    forest_class1.fit(X_train, y_train)
    X_test1 = ["Junior", "Python", "no", "no"] # Should be True
    y_predicted1 = forest_class1.predict(X_test1)
    assert y_predicted1 == forest_class1_predict_answer

    # Test 2: N = 10, M = 7, and F = 2
    forest_class2 = MyRandomForestClassifier(10, 7, 2)
    forest_class2.fit(X_train, y_train)
    X_test2 = [["Mid", "Python", "no", "no"]] # Should be True
    y_predicted2 = forest_class2.predict(X_test2)
    assert y_predicted2 == forest_class2_predict_answer
    
    # Test 3: N = 10, M = 3, and F = 1
    forest_class3 = MyRandomForestClassifier(10, 3, 2)
    forest_class3.fit(X_train, y_train)
    X_test3 = [["Senior", "R", "yes", "no"]] # Should be True
    y_predicted3 = forest_class3.predict(X_test3)
    assert y_predicted3 == forest_class3_predict_answer
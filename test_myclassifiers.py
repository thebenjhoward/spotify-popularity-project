from altsklearn.myclassifiers import MyDecisionTreeClassifier


## Decision Tree Models ##
# interview dataset
interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
interview_table = [
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

interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

interview_tests = [
    ["Junior", "Java", "yes", "no"],
    ["Junior", "Java", "yes", "yes"]
]

interview_true = [ "True", "False" ]

# bramer degrees dataset
degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

degrees_tree = \
    ["Attribute", "att0", # SoftEng
        ["Value", "A",
            ["Attribute", "att4", # Project
                ["Value", "A",
                    ["Leaf", "FIRST", 5, 14]
                ],
                ["Value", "B", 
                    ["Attribute", "att3", # CSA
                        ["Value", "A",
                            ["Attribute", "att1", # ARIN
                                ["Value", "A",
                                    ["Leaf", "FIRST", 1, 2]
                                ],
                                ["Value", "B",
                                    ["Leaf", "SECOND", 1, 2]
                                ]
                            ]
                        ],
                        ["Value", "B", 
                            ["Leaf", "SECOND", 7, 9]
                        ]
                    ]
                ]
            ]
        ],
        ["Value", "B",
            ["Leaf", "SECOND", 12, 26]
        ]
    ]

degrees_tests = [
    ["B", "B", "B", "B", "B"],
    ["A", "A", "A", "A", "A"],
    ["A", "A", "A", "A", "B"]
]

degrees_true = [ "SECOND", "FIRST", "FIRST"]

def test_decision_tree_classifier_fit():

    # Split off the last column from the rest of table
    *X_cols, y_interview = list(map(list, zip(*interview_table)))
    X_interview = list(map(list, zip(*X_cols)))

    interview_model = MyDecisionTreeClassifier()
    interview_model.fit(X_interview, y_interview)
    
    assert interview_model.tree == interview_tree


    *X_cols, y_degrees = list(map(list, zip(*degrees_table)))
    X_degrees = list(map(list, zip(*X_cols)))

    degrees_model = MyDecisionTreeClassifier()
    degrees_model.fit(X_degrees, y_degrees)

    assert degrees_model.tree == degrees_tree



def test_decision_tree_classifier_predict():
    
    # we just set the value of tree here to avoid having to re-prep the data
    interview_model = MyDecisionTreeClassifier()
    interview_model.tree = interview_tree
    
    assert interview_model.predict(interview_tests) == interview_true
    
    degrees_model = MyDecisionTreeClassifier()
    degrees_model.tree = degrees_tree

    assert degrees_model.predict(degrees_tests) == degrees_true

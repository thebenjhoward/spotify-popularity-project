##############################################
# Programmer: Ben Howard and Elizabeth Larson
# Class: CPSC 322-01, Spring 2021
# Final project
# 05/05/21
#
# Sources:
#       Checking if all list values are the same (case 2 decision trees): https://www.geeksforgeeks.org/python-check-if-all-elements-in-a-list-are-same/
#       Splitting a string with split() (printing association rules): https://www.techiedelight.com/split-string-into-list-python/
# 
# Description: This program computes reusable general-purpose functions. It also
#              includes a header where PA utils end and project utils begin.
##############################################

# TODO: Finish all TODOs

import random # For majority voting leaf node "flip a coin" solution (if the clash result is 50%/50%)
import math # For log calculations
import itertools # For subset building

def compute_euclidean_distance(v1, v2):
    """Calculate the euclidean distance between two vectors

    Args:
        v1(list of numeric vals): First vector
        v2(list of numeric vals): Second vector
        
    Returns:
        dist(float): The distance between the vectors
    """
        
    # Just look at the first two points in the vectors
    new_v1 = [v1[0], v1[1]]
    new_v2 = [v2[0], v2[1]]
        
    assert len(new_v1) == len(new_v2)
    dist = (sum([(new_v1[i] - new_v2[i]) ** 2 for i in range(len(new_v1))])) ** (1/2) # Get the square root by taking this formula to the 1/2 power
    return dist

def convert_to_DOE(values):
    """Convert a list of values (MPG, for this dataset) to the DOE values listed in the table in step 1's notebook/write-up
    
        Args:
            values (list of float): The values were are converting
            
        Returns:
            converted_values (list of int): These conversted values on a scale of [1-10]
    """
    converted_values = []
    
    for value in values:
        if value <= 13: # ≤13
            converted_values.append(1)
        elif value > 13 and value <= 14: # 14
            converted_values.append(2)
        elif value > 14 and value <= 16: # 15-16
            converted_values.append(3)
        elif value > 16 and value <= 19: # 17-19
            converted_values.append(4)
        elif value > 19 and value <= 23: # 20-23
            converted_values.append(5)
        elif value > 23 and value <= 26: # 24-26
            converted_values.append(6)
        elif value > 27 and value <= 30: # 27-30
            converted_values.append(7)
        elif value > 30 and value <= 36: # 31-36
            converted_values.append(8)
        elif value > 36 and value <= 44: # 37-44
            converted_values.append(9)
        elif value < 44: # ≥45
            converted_values.append(10)
            
    return converted_values

def normalize_data(values):
    """Normalize a group of values to a 0.0-1.0 scale

    Args:
        values(list of obj): Data we want to normalize
        
    Returns:
        noramlized_values(float): These values after calulations
    """
    
    normalized_values = []
    
    # value - smallest value in the dataset
    min_value = min(values)
    for index in range(len(values)):
        normalized_values.append(values[index] - min_value)

    # value / largest value in the dataset
    max_value = max(normalized_values)
    for index in range(len(normalized_values)):
        normalized_values[index] = normalized_values[index] / max_value
    
    return normalized_values

def calculate_accuracy_and_error_rate(matrix):
    """Uses a confusion matrix to determine the amount of correct and incorrect guesses
    Use these values to compute the accuracy and error rate of the matrix

    Args:
        matrix(list of list of obj): The confusion matrix we're checking
        
    Returns:
        accuracy(float): How many guesses were correct (decimal form of %)
        error_rate(float): How many guesses were incorrect (decimal form of %)
    """
    
    # Add up all values in the datasets
    total = 0.0
    for value in matrix:
        for value_index in range(len(value)):
            total += value[value_index]

    if total != 0.0: # Only do this calulating if there was at least one correct prediction
        # Keep track of the correctly guessed ones (where actual is 1 and pred is 1 and so on)
        # Also keep track of incorrect guesses: times when the predicted guessed
        correct_guesses_total = 0
        incorrect_guesses_total = 0
        for row_index in range(len(matrix)):
            for col_index in range(len(matrix[row_index])):
                if (row_index + 1) == (col_index + 1): # e.g. row_index=0 and col_index=0 would be the pairing for predicting 1 and being right... the diagonal from 0,0 to N,N on the matrix
                    correct_guesses_total += matrix[row_index][col_index]
                    break # Now stop checking the cols and go to the next row
                elif matrix[row_index][col_index] != 0: # Skip 0 values because these aren't predictions
                    incorrect_guesses_total += matrix[row_index][col_index]

        # Now, calculate the accuracy and error rate
        accuracy = correct_guesses_total / total
        error_rate = incorrect_guesses_total / total
    else: # Nothing was correct
        accuracy = 0.0
        error_rate = 1.0
    
    return accuracy, error_rate

def calculate_distance_categorical_atts(X, y, v):
    """Calculate the predicting class of a vector that's categorical

    Args:
        X(list of list of obj): X_train (the dataset)
        y(list of obj): y_train (col we're predicting on)
        v(list of numeric vals): Vector values
        
    Returns:
        dist(obj): The predicted value
    """
    
    # Go through each row in X and find the "closest" value (i.e. the attribute with the most matching values)
    num_matching_atts = []
    for row_index in range(len(X)):
        matching_atts_count = 0
        for col_index in range(len(v)):
            if v[col_index] == X[row_index][col_index]: # Found a match!
                matching_atts_count += 1
        num_matching_atts.append(matching_atts_count)
            
    # Find the row that has the most matches on it
    row_with_most_matching_atts = num_matching_atts.index(max(num_matching_atts))
    dist = y[row_with_most_matching_atts]
    
    return dist

def all_same_class(instances):
    """Check if all instance labels match the first label

    Args:
        instances(list of lists): Instance set we're checking
        
    Returns:
        True or False, depending on if all of the instance labels match the first label
    """

    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # Otherwise, all instance labels matched the first label

def select_attribute(instances, available_attributes, domains, y_train_col_index):
    """Pick an attrubite to split on based on entropy calculations

    Args:
        instances(list of lists): Instance set we're calculating the entropy of
        available_attributes(list): Potential attributes we can split on
        domains(dict): Possible values for each col (e.g. "yes" and "no")
        y_train_col_index(int): Col index of the y_train attribute (not for splitting)
        
    Returns:
        available_attributes[att_to_split_on_index](string): The name of the attribute we're splitting on
    """

    # Calculate the attribute domains dictionary (e.g. standing can be 1 or 2)
    e_news = []
    domains_list = list(domains.items())
    for index in range(len(available_attributes)):
        if y_train_col_index != index: # Skip the att we're trying to predict on (e.g. interviews_well)
            e_news.append(calculate_e_new(index, y_train_col_index, instances, domains_list[index], domains_list[y_train_col_index]))

    # Choose the smallest of the four and split on that, but also check for duplicate e_news calculations (occurs in the interview test dataset)
    try:
        att_to_split_on_index = e_news.index(min(e_news))
        for e_new_index in range(len(e_news)):
            if (e_new_index + 1) < len(e_news):
                if e_news[e_new_index] == e_news[e_new_index + 1] and e_new_index == att_to_split_on_index:
                    att_to_split_on_index = e_new_index + 1
    except ValueError: # For when e_news is empty
        att_to_split_on_index = 0

    return available_attributes[att_to_split_on_index]

def partition_instances(instances, split_attribute, headers, domains):
    """Break up a set of instances into partitions using the split attribute

    Args:
        instances(list of lists): Instance set we're partitioning
        split_attribute(string): Attribute name we're going to be splitting on
        headers(list): Attribute names, corresponds with the instances
        domains(dict): Possible values for each col (e.g. "yes" and "no")
        
    Returns:
        partitions(dict): Partitions organized by attibute value (the key)
    """
    
    # Comments refer to split_attribute "level" in the interview test set
    attribute_domain = domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = headers.index(split_attribute) # 0

    partitions = {} # key (attribute value): value (partition)
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
                
    return partitions

def tdidt(current_instances, available_attributes, headers, domains):
    """Create a tree given a set of instances
    Handles 3 cases (listed in the comments below)

    Args:
        current_instances(list of lists): Instances we're looking at
        available_attributes(list): Attribute names we can still split on
        headers(list): All attribute names
        domains(dict): Possible values for all atts
        
    Returns:
        A constructed tree (as a list of lists of lists...)
    """
    
    # Select an attribute to split on, then remove if from available attributes
    split_attribute = select_attribute(current_instances, available_attributes, domains, (len(available_attributes) - 1))
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute]

    # Group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, headers, domains)

    # For each partition, repeat unless one of the following base cases occurs:
    #   CASE 1: All class labels of the partition are the same => make a leaf node
    #   CASE 2: No more attributes to select (clash) => handle clash w/majority vote leaf node
    #   CASE 3: No more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        if len(partition) > 0 and all_same_class(partition): # Case 1
            leaf_subtree = ["Leaf", partition[-1][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf_subtree)
        elif len(partition) > 0 and len(available_attributes) == 0: # Case 2
            leaf_value = perform_majority_voting(current_instances, domains) # Perform majority voting
            # Create a leaf node based on this
            leaf_subtree = ["Leaf", leaf_value, len(partition), len(current_instances)]
            values_subtree.append(leaf_subtree)
        elif len(partition) == 0: # Case 3
            leaf_value = perform_majority_voting(current_instances, domains) # Perform majority voting
            # Create a leaf node based on this
            leaf_subtree = ["Leaf", leaf_value, len(current_instances), len(current_instances)]
            values_subtree.append(leaf_subtree)
        else: # All base cases are false... time to recurse!
            subtree = tdidt(partition, available_attributes.copy(), headers, domains)
            values_subtree.append(subtree)
        tree.append(values_subtree)
    return tree

def perform_majority_voting(clashing_instances, domains):
    """Look for a leaf node value for clashing instances
    Looks for the value that occurs most frequently in the dataset
    If it'd an even split, flip a coin (pick a number 0-the length of the y_train domain)

    Args:
        clashing_instances(list of lists): The instances we're looking for the leaf node value of
        domains(dict): Possible values for each col (e.g. "yes" and "no")
        
    Returns:
       The leaf node value
    """

    is_even_split = True

    # What is our most popular label in this clash?
    domains_list = list(domains.items())
    possible_domain_values = domains_list[-1][1]
    domain_counts = [] # Parallel to possible_domain_values
    for domain_value in possible_domain_values:
        count = 0
        for value in clashing_instances:
            if value[-1] == domain_value:
                count += 1
        domain_counts.append(count)

    # Check if all of the counts are the same (if so, we have an even_split... and if not, find the )
    count_to_check = domain_counts[0]
    for count in domain_counts:
        if count_to_check != count:
            is_even_split = False
            break # Stop searching! We found a difference

    if is_even_split: # Both the same? 50/50? Flip a coin
        coin = random.randint(0, len(possible_domain_values) - 1) # Random number 0-the number of values in y_train domain (e.g. "yes" and "no would be 0-1")
        return possible_domain_values[coin]
    else: # Otherwise, return the value that occurs the most
        index_of_largest_domain_count = domain_counts.index(max(domain_counts))
        return possible_domain_values[index_of_largest_domain_count]

def calculate_entropy(priors, entropy_values):
    """Calculate weighted average of partition entropies
    Priors and entropy values are parallel lists

    Args:
        priors(list): Total occurances in the dataset
        entropy_values(list): Calculated entropy values
        
    Returns:
        avg(int): Average of the priors and entropy values
    """
    
    avg = 0.0
    for i in range(len(entropy_values)):
        avg = avg + (priors[i] * entropy_values[i])
    return avg
    
def calulate_entropy_for_one_partition(values):
    """Calculate the entropy of a partition given values

    Args:
        values(list): The values we're calculating the entropy of
        
    Returns:
        e(float): Calculated entropy
    """

    e = -(values[0] * math.log(values[0], 2))
    index = 1 # Start at index 1 since we've already saved [0] in e
    while index < len(values):
        e = e - (values[index] * math.log(values[index], 2))
        index += 1
    return e
  
def calculate_e_new(col_index, y_train_col_index, instances, domain, y_train_domain):
    """Calculate entropy stats for a domain (priors, entropy for each, total entropy)

    Args:
        col_index(int): Col index of the att we're calulating the entropy of
        y_train_col_index(int): y_train col index
        instances(list of lists): The data table
        domain(dict): Possible values of the att we're calulating the entropy of
        y_train_domain(dict): Possible values of the y_train value
        
    Returns:
        e_new(float): Total entropy
    """
    
    # Find the total number of instances in the dataset
    total = len(instances)
    
    # Load priors (aka how many times do Senior/Mid/Junior appear total?)
    priors = []
    for domain_value in domain[1]: # domain[1] gives a list of domain values
        count = 0
        for instance in instances:
            if instance[col_index] == domain_value:
                count += 1
        priors.append(count/total)
    
    # Entropy of the each domain value (e.g. e of Senior, Mid, and Junior for level)
    # Check for matches (e.g. all cases of Senior and False, then Senior and True...)
    entropy_values = []
    for domain_value in domain[1]:
        values_to_calc = []
        for y_train_domain_value in y_train_domain[1]:
            count = 0
            total = 0
            for instance in instances:
                if instance[col_index] == domain_value:
                    if instance[y_train_col_index] == y_train_domain_value:
                        count += 1 # Both values match! Incremeant the count (numerator)
                    total += 1 # Either way, incremeant the total (denominator)
            if total == 0:
                values_to_calc.append(0.0)
            else:
                values_to_calc.append(count/total)
        
        try:
            e = calulate_entropy_for_one_partition(values_to_calc)
        except ValueError: # For when the calc is undefined
            e = 0.0
        entropy_values.append(e)
        
    # Weighted average of its partition entropies
    e_new = calculate_entropy(priors, entropy_values)
    return e_new

def predict_recursive_helper(tree, X_test):
    """Predict the leaf node based on X_test values
    Handles cases where the y_test attribute is split on

    Args:
        tree(list of lists): The tree we're checking
        X_test(list of lists): Values we're predicting for (subtree that doesn't include the attibute we're predicting)
        
    Returns:
        Either the leaf value or a recursive call to this function
    """
    
    label = tree[0] # e.g. Attribute, Value, or Leaf
    if label == "Attribute":
        # Get the index of this attribute using the name (i.e. att0 is at index [0] in the attribute names)
        att_index = 0 # Default is 0
        for letter in tree[1]:
            if letter != "a" and letter != "t":
                att_index = letter
        att_index = int(att_index)
        
        # In case we split on the class label
        if att_index >= len(X_test):
            return tree[2][2][1]
        
        # Grab the value at this index and see if we have a match going down the tree
        instance_value = X_test[att_index]
        i = 2
        while i < len(tree):
            value_list = tree[i]
            if value_list[1] == instance_value: # Recurse when a match is found
                return predict_recursive_helper(value_list[2], X_test)
            i += 1
    else:
        return tree[1] # Grab the value of the leaf

def make_rule_for_one_branch(tree, attribute_names, class_name, rule):
    """Grab a list of strings that represents one branch's rule (see args for formatting)
    Assumes that ruleis already populated with the split attribue info (["IF", "att0", "=", "value"]) upon initial call
    
    Args:
        tree(list of lists): The tree/subtree we're looking at
        attribute_names(list of str or None): A list of attribute names to use in the decision rules
            (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
        class_name(str): A string to use for the class name in the decision rules
            ("class" if a string is not provided and the default name "class" should be used).
        rule(list of strings): A rule, formatted like IF att0 == value AND ... THEN class_label = True (but with each element in an encapsulating list)

    Returns:
        list of strings: The full branch's rule
    """
    
    label = tree[0] # e.g. Attribute, Value, or Leaf
    if label == "Leaf": # We've hit the end of a branch
        rule.append("THEN")
        rule.append(class_name)
        rule.append("==")
        rule.append(tree[1]) # The value of the leaf
        return rule
    elif label == "Attribute":
        rule.append("AND")
        if attribute_names == None:
            rule.append(tree[1])
        else: # [-1]st spot of att labels is the index
            att_index = int(tree[1][-1])
            rule.append(attribute_names[att_index])
        rule.append("==")

        # There will be more to the initial tree beyond the leaf we run into here, because of this attribute split (e.g. one rule where phd = yes and one where phd = no)
        index = 2 # Values start at index 2
        new_rules = []
        
        while index < len(tree): # Go through the values on each partition (e.g. Junior, Mid and Senior)
            # Calculate the branch (initial attribute that's already in there is passed in as rule)
            new_rule = make_rule_for_one_branch(tree[index], attribute_names, class_name, rule)

            new_rules.append(new_rule)
            index += 1
            if index < len(tree): # Check if we've hit the end of the tree (and if so, don't add any more rules)
                rule = []
                if attribute_names == None:
                    rule = [tree[1], "=="]
                else: # [-1]st spot of att labels is the index
                    att_index = int(tree[1][-1])
                    rule = [attribute_names[att_index], "=="]

        return new_rules
    else: # Otherwise, it's a value
        rule.append(tree[1])
        return make_rule_for_one_branch(tree[2], attribute_names, class_name, rule) # Recurse on subtree

def compute_unique_values(table):
    """Go through each item in the table and keep track of unque values

    Args:
        table (list of lists): Entire dataset
        
    Returns:
        unique (set): Possible values/domain of the dataset
    """

    unique = set()
    for row in table:
        for value in row:
            unique.add(value)
    return sorted(list(unique))

def apriori(table, minsup, minconf):
    """Produce a set of itemsets that are all well-supported
    Take this set and generate rules

    Args:
        table (list of lists): Entire dataset
        minsup (float): Minimum support, for supported itemsets
        minconf (float): Minimum confidence, evaluating supported itemsets
        
    Returns:
        rules (list of strings): Rules written in IF/THEN form
    """

    all_itemsets = []
    supported_itemsets = []
    
    # Create L(1) (which is a set of supported itemsets of cardinality 1)
    # Cardinality = size of the set, so L(1) is a set where each element is of size 1
    L = compute_unique_values(table)
    # How many transactions match "b"? How many match "c"? ...
    matching_calcs = [] # Parallel to L(1)
    for L_value in L:
        count = 0
        for row in table:
            for row_value in row:
                if row_value == L_value:
                    count += 1
                    break
                # Otherwise, keep searching
        matching_calcs.append(count/len(table))
    # Use these results to prune the dataset (aka only save stuff larger than or equal to minsup)
    pruned_L = []
    for i in range(len(matching_calcs)):
        if matching_calcs[i] >= minsup:
            pruned_L.append(L[i])
    L = pruned_L

    # Set k to 2 (k is cardinalty/size of the dataset)
    k = 2

    # While L(k-1) isn't an empty set...
    while L != []:
        # Create C(k) from L(k-1)
        # Join Step:
        #   Compare each member of L(k-1) with every other member
        #   If the first k-2 items are identical, put A ∪ B in C(k)
        Ck = []
        if k == 2: # 2-2 = 0, so no need to check for A ∪ B stuff
            for value in L:
                value_index = L.index(value)
                for value2 in L:
                    value2_index = L.index(value2)
                    if value2 != value and value2_index >= value_index: # So we don't get, say, {b, b}
                        Ck.append([value, value2])
        else: # Otherwise, k-2 > 0
            A_index = 0
            while A_index < len(L) - 1: # -1 so we skip the last element (how do you check if {e, s} matches the next element if there is no next element?)
                B_index = A_index + 1
                items_to_check = k - 2
                while B_index < len(L):
                    is_match = False
                    i = 0
                    #i = items_to_check - 1 
                    while i <= items_to_check - 1: # -1 because Python indecies are 0-based, so checking the first element would be [0]
                        if L[A_index][i] == L[B_index][i]:
                            is_match = True
                        i += 1
                        #i -= 1
                    if is_match == True: # It's a full match, so add A ∪ B to C(k)
                        Ck.append(find_union(L[A_index], L[B_index]))
                    B_index += 1
                A_index += 1
        # Prune Step:
        #   For each member of Ck, check all subsets of c with k-1 elements
        #   Delete c from Ck if any of the subsets is not a member of L(k-1)
        for c in Ck:
            subset = []
            subset.extend(itertools.combinations(c, k - 1)) # Comes back as a list of tuples
            # Convert to list values
            list_subset = []
            for subset_value in subset:
                list_subset.append(list(subset_value))
            subset = list_subset
            # Check if this subset is in L(k-1)
            for subset_value in subset:
                was_found = False
                for L_value in L:
                    if k == 2:
                        if subset_value[0] == L_value:
                            was_found = True
                            break # We found it in L(k-1)! Stop searching and go to the next subset value
                    else:
                        if subset_value == L_value:
                            was_found = True
                            break # We found it in L(k-1)! Stop searching and go to the next subset value
                if was_found == False: # Time to prune!
                    try:
                        Ck.remove(c)
                    except ValueError:
                        break
                    if Ck == []: # Done pruning!
                        break

        # Prune all itemsets in C(k) that are not support to create L(k)
        Lk = []
        for c in Ck:
            count = 0
            # How many times is {b, c} a subset of transaction? How about {b, m}?
            for row in table:
                row_subsets = []
                row_subsets.extend(itertools.combinations(row, k)) # Comes back as a list of tuples, so convert to list down the road
                for subset in row_subsets:
                    if c == list(subset):
                        count += 1 # We found a match!
            # Only keep the c we're looking at if it meets minimal support
            if count/len(table) >= minsup:
                Lk.append(c)
        # Now, each item in L(k) has cardinality k
        # So replace our original L with this new one and keep on trucking along
        L = Lk
        all_itemsets.append(L)

        # Increase k by 1
        k += 1

    # Add L(2) and L(k-2) to supported_itemsets
    supported_itemsets_seperated = []
    for itemset in all_itemsets:
        index = all_itemsets.index(itemset) + 2
        if index == 2 or index == (k - 2):
            supported_itemsets_seperated.append(itemset)
    # Now smush them together into one big itemset list
    for L in supported_itemsets_seperated:
        for itemset in L:
            supported_itemsets.append(itemset)

    # Generate rules
    rules = generate_apriori_rules(supported_itemsets, table, minconf)
    
    return rules

def generate_apriori_rules(supported_itemsets, table, minconf):
    """Generate confident rules in an IF/THEN format

    Args:
        supported_itemsets (list of lists): Supported itemsets to check
        table (list of lists): Entire dataset
        minconf (float): Confidence level we're checking against
        
    Returns:
        rules (list of strings): Rules written in IF/THEN form
    """
    
    rules = []

    for itemset in supported_itemsets:
        rhs_size = 1
        while rhs_size <= len(itemset) - 1: # -1 to ensure the LHS always has at least 1 term
            # Find possible lhs and rhs vlaues
            lhs, rhs = calculate_lhs_and_rhs(itemset, rhs_size)

            # Prune any rules that don't have confidence that is >= to the minconf value we passed in
            confidence_calcs = calculate_confidence(lhs, rhs, itemset, table)
            
            # Choose which rules to keep based on calculations
            for index in range(len(confidence_calcs)):
                if confidence_calcs[index] >= minconf:
                    # Generate lhs string
                    i = 0
                    lhs_string = ""
                    while i < len(lhs[index]):
                        lhs_string += str(lhs[index][i])
                        i += 1
                        if i < len(lhs[index]): # We're not at the end yet, so add an AND
                            lhs_string += " AND "
                    # Generate rhs string
                    j = 0
                    rhs_string = ""
                    while j < len(rhs[index]):
                        rhs_string += str(rhs[index][j])
                        j += 1
                        if j < len(rhs[index]): # We're not at the end yet, so add an AND
                            rhs_string += " AND "
                    new_rule = "IF " + lhs_string + " THEN " + rhs_string
                    rules.append(new_rule)

            rhs_size += 1

    return rules

def calculate_lhs_and_rhs(itemset, rhs_size):
    """Finds the lhs (after the IF and before the THEN) and rhs (after the THEN) values of an itemset

    Args:
        itemset (list): Supported itemset being checked
        rhs_size (int): How many elements are in the rhs list
        
    Returns:
        lhs and rhs (list of lists): Left-hand side and right-hand side values
    """

    lhs = []
    rhs = []

    # All possible rhs_size-term RHSs
    rhs_tuples = []
    rhs_tuples.extend(itertools.combinations(itemset, rhs_size)) # Comes back as a list of tuples, so convert to list down the road
    for value in rhs_tuples:
        rhs.append(list(value))

    # Looking at each item in RHS, all un-used items in L ∪ R save in the corresponding LHS
    for rhs_item in rhs:
        mini_lhs = []
        for item in itemset:
            if not item in rhs_item:
                mini_lhs.append(item)
        lhs.append(mini_lhs)

    return lhs, rhs

def calculate_confidence(lhs, rhs, itemset, table):
    """Find the confidence of a rule

    Args:
        lhs (list of lists): Left hand side of the rule
        rhs (list of lists): Right hand side of the rule
        itemset (list): Supported itemset being checked
        table (list of lists): The entire dataset
        
    Returns:
        confidence_calcs (list of floats): Calculated confidences of each rule (on a scale of 0.0-1.0 (aka 0%-100%))
    """

    confidence_calcs = [] # count(L ∪ R)/count(L)
    
    # Count number of subset matches in the original dataset (numerator)
    num_count = 0
    for row in table:
        subsets = []
        subsets.extend(itertools.combinations(row, len(itemset))) # Comes back as a list of tuples, so convert to list down the road
        for subset in subsets:
            if list(subset) == itemset:
                num_count += 1
    
    # All of the numerators are the same for an itemset, so we only need to calucalate this once
    # Count number of LHS matches in the original dataset (denominator)
    for lhs_value in lhs:
        den_count = 0
        for row in table:
            subsets = []
            subsets.extend(itertools.combinations(row, len(lhs_value))) # Comes back as a list of tuples, so convert to list down the road
            for subset in subsets:
                if list(subset) == lhs_value:
                    den_count += 1
        if den_count == 0: # So we aren't dividing by 0
            confidence_calcs.append(0.0)
        else:
            confidence_calcs.append(num_count/den_count)

    return confidence_calcs

def find_union(A, B):
    """Find the union of two sets

    Args:
        A (list): Set we're checking
        B (list): Set we're checking
        
    Returns:
        Union (list): A ∪ B
    """
    
    union = []

    # Add all A values to union
    for A_value in A:
        union.append(A_value)

    # Add all B values to union, skipping them if they're already in there (aka matching values in A and B)
    for B_value in B:
        if not B_value in union:
            union.append(B_value)

    return union

def calculate_rule_interestingness(rule, table):
    """Calculate support, confidence, and lift of a given rule

    Args:
        rule (string): List we're looking at
        table (list of lists): Entire dataset
        
    Returns:
        support (float): Support of said rule (0.0-1.0 scale)
        confidence (float): Confidence of said rule (0.0-1.0 scale)
        lift (float): Lift of said rule (0.0-1.0 scale)
    """

    support = 0.0
    confidence = 0.0
    lift = 0.0
    lhs = []
    rhs = []

    # Grab the lhs and rhs from the rule
    rule_list = rule.split(' ')
    in_rhs = False
    for rule_value in rule_list:
        if rule_value == "THEN": # We're officially on the right-hand side
            in_rhs = True
        elif rule_value != "IF" and rule_value != "AND":
            if in_rhs == True:
                rhs.append(rule_value)
            else:
                lhs.append(rule_value)

    # Calculate support
    Nboth = 0
    for row in table:
        is_in_row = False
        for lhs_value in lhs:
            if lhs_value in row:
                is_in_row = True
            else:
                is_in_row = False
        if is_in_row == True: # Only search rhs if we know that lhs is in there
            for rhs_value in rhs:
                if rhs_value in row:
                    is_in_row = True
                else:
                    is_in_row = False
            if is_in_row == True:
                Nboth += 1
    Ntotal = len(table)
    support = Nboth / Ntotal

    # Calculate confidence
    Nleft = 0
    for row in table:
        is_in_row = False
        for lhs_value in lhs:
            if lhs_value in row:
                is_in_row = True
            else:
                is_in_row = False
        if is_in_row == True:
            Nleft += 1
    confidence = Nboth / Nleft

    # Calculate lift
    lift = support / ((support) * (support))

    return support, confidence, lift



"""------------------------------------------------------------
-------------- Project Util Functions Begin Here --------------
------------------------------------------------------------"""
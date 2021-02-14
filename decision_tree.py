""" 
    decision_tree.py - CMPT 459 - Jong Joon Lee 
        - a program to grow a decision tree based on test data
        - trains adult.data.csv and tests adult.test.data.csv
        - prunes the decision tree for error-reduction
        - grow(dataset) returns the tree object based on the dataset
        - prune(data, tree) returns the pruned tree object
        - test(data, tree) returns the accuracy of the test data based on the training data
        - output(tree) outputs the computed tree into a csv file called predictions.csv
    """

import numpy as np
import pandas as pd

training_data = pd.read_csv("adult.data.csv")
test_data = pd.read_csv("adult.test.csv")

validation_data_percentage = 0.1   # 10% of training data can be used as validation data

attributes = [
    'age', 'workclass', 'fnlwgt','education', 
    'education-num', 'marital-status',
    'occupation','relationship', 'race',
    'sex', 'capital-gain', 'capital-loss', 
    'hours-per-week', 'native-country', 'income'
]
numerical_attribute_index = [0,2,4,10,11,12]
categorical_attribute_index = [1,3,5,6,7,8,9,13,14]

# Returns true if end condition of a single column is met
def is_unique_label(data):
    # Store columns of the remaining data frame
    remaining_cols = data.columns
    if len(remaining_cols) == 1:
        return True
    else:
        return False

# Returns the most common data in the remaining values in the single column
def data_result(data):
    col_name = data.columns[0]
    mode_value = data[col_name].mode()
    x = mode_value[0]
    return str(x)

# Finds the entropy of an array or dataframe
def entropy(array):
    # entropy = - sum of probability of pi log2 pi
    #print(c)
    if(type(array) == list):
        unique_array, counter_of_unique_array = np.unique(array, return_counts=True)
    else:
        unique_array, counter_of_unique_array = np.unique(array.flatten().tolist(), return_counts=True)
    probability_array = (counter_of_unique_array)/(np.sum(counter_of_unique_array))
    #print("prob array:", probability_array)
    total = 0
    for p in probability_array:
        total += (p)*-(np.log2(p))
    #print("total", total)
    return total

# Calculates the information gain based on the given data and threshold to divide the data 
def information_gain(data, column, threshold, col_num):
    # infomration_gain = Entropy(T) - [weighted averages] * Entropy(Ti)
    entropy_T = entropy(data)
    #print(entropy_T)
    # use the threshhold to split the information into left and right partitions
    if(col_num in categorical_attribute_index):
        left_partition = np.argwhere(column != threshold)
        right_partition = np.argwhere(column == threshold)
    else:
        left_partition = np.argwhere(column <= threshold)
        right_partition = np.argwhere(column > threshold)
    # use the weighted average by length of each partition * entropy of the parition
    length_left = len(left_partition.flatten())
    length_right = len(right_partition.flatten())
    if(length_left == 0 or length_right == 0): 
        return 0     # entropy T will be same as sum of entropy Ti
    left_values = []
    right_values = []

    for x in left_partition.flatten(): 
        left_values.append(column[x])
    for y in right_partition.flatten(): 
        right_values.append(column[y])

    length_total = length_left + length_right
    left_entropy = (length_left/length_total)*entropy(left_values)
    right_entropy = (length_right/length_total)*entropy(right_values)
    #print("info gain:", entropy_T - (left_entropy+right_entropy))
    return entropy_T - (left_entropy+right_entropy)

# Returns two seperate data sets partition by the decision made by calculating the information gain by a greedy search 
def learnDecision(data):
    best_gain = -9999
    data_values = data.values
    rows_count, cols_count = data_values.shape
    for cols in range(cols_count):
        thresholds = np.unique(data_values[:,cols])
        #print("thresholds: ", thresholds)
        for threshold in thresholds:
            #print("current threshold: ", threshold)
            info_gain =  information_gain(data_values[:,cols], data_values[:,cols], threshold, cols)
            #print(info_gain)
            if info_gain > best_gain:
                best_gain = info_gain
                split_col = cols
                split_threshold = threshold
    # if it is a categorical attribute
    if(split_col in categorical_attribute_index):
        left_partition = np.argwhere(split_col == threshold)
        right_partition = np.argwhere(split_col != threshold)
    else:
        left_partition = np.argwhere(split_col <= threshold)
        right_partition = np.argwhere(split_col > threshold)
    return left_partition, right_partition, split_col, split_threshold

# Recursively builds the decision tree based on the given data
def grow(data):
    ## Base Case: Only one unique label is left
    if (is_unique_label(data) == True):
        ## format data in an array 
        leaf_node = {"values":data_result(data)}
        return leaf_node
    
    ## Recursive case: If we have more than one type of label  
    else:
        left_data, right_data, split_col, split_threshold = learnDecision(data) 
        type = 'numerical' if split_col in numerical_attribute_index else 'categorical'
        node = { 'question':attributes[split_col],
                 'threshold': split_threshold,
                 'type': type}
        left_node = grow(left_data)
        right_node = grow(right_data)
        node['left node'] = (left_node)
        node['right node'] = (right_node)
        return node

# Returns the accuracy of the given tree on a given dataset
def test(data, tree):
    test_data_values = data.values
    total_correct = 0
    total = 0
    for entries in test_data_values:
        if test_data_values[:,entries] in tree:
            if tree[test_data_values[entries]] == True:
                total_correct += 1 
                total += 1
            else: 
                total += 1
    np_total_correct = np.float32(total_correct)
    np_total = np.float32(total)
    # Accuracy is the total correct / total 
    return (np_total_correct/np_total)

# Prunes the tree object using the validation dataset recursively
def prune(data, subtree, error):
    validation_index = data.values
    for rows in range(validation_index.shape[1]):
        # Save possible trees to prune
        data = validation_index[rows]
        # Need to calculate the change in error if we remove certain trees
        for columns in validation_index: 
            pruned_tree = (validation_index[columns], subtree)
            minimuim_error = -(test(subtree))
            if error > minimuim_error:
                return prune(data, pruned_tree, error)
        # Minimum error subtree 
        return subtree

# Outputs the computed tree into a csv file called predictions.csv
def output_xsv(tree):
    tree_df = pd.DataFrame(data=tree)
    tree_df.to_csv('predictions.csv')
    return




# Decision Tree Implementation in Python

The pythonic implementation of a decision tree classifier that is able to preprocess data, recursively built a tree and prune the resultant tree. 
[Learn more about decision tree learning.](https://en.wikipedia.org/wiki/Decision_tree_learning)

## Installation

To install the required packages: 

On macOS/Linux:
```
python -m pip install -r requirements.txt
```
On Windows:
```
py -m pip install -r requirements.txt
```

Run the python file in your favorite editor or by this command: 
```
python3 decision_tree.py
```


## Predictive Classifiers & Decision Trees

Decision tree learning is one of the most practical and predictive modelling techniques used in data mining, statistics and machine learning. By building a decision tree with branches that represent possible paths or options observed by an item which lead to leaf nodes that represent conclusions about the target value, a decision tree is simple to understand and practical in its usage. As a tree structure, it is also used to visually represent the decisions and decision making by the predictive model. 

The accuracy of the model is highly depended on the given dataset. This current implementation requires a separate datasets to build and test the model rather than using the same dataset to do so. In our above data, our decision tree is capable of predicting the income of a person mainly using the age, work, education, marital status and ethnicity features of the data entry. 

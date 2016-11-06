#! /usr/bin/env python3

import numpy as np
from sklearn.datasets import load_iris #http://scikit-learn.org/stable/datasets/
from sklearn import tree

#import pdf graphing
from sklearn.externals.six import StringIO
import pydotplus

iris = load_iris()

#print(iris.feature_names)   #attributes i.e. petal width
#print(iris.target_names)    #types i.e. setosa (a flower species)

#print(iris.data[0]) #prints values for all features in the features table of index 0
#print(iris.target[0]) #prints target element position

#print all elements in the list of iris
#for i in range(len(iris.target)): #for each target value
#    print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))
'''
Recoginising flowers based on prior measurements
'''

#TRAINING
test_idx        =   [0,50,100]
train_target    =   np.delete(iris.target, test_idx) #delete 3 vars
train_data      =   np.delete(iris.data, test_idx, axis=0)

#TESTING
test_target     =   iris.target[test_idx]
test_data       =   iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target) # clf.fit(features, labels)

print(test_target)
print(clf.predict(test_data))


#PRINT OUT DECISION TREE
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        impurity=False
                    )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

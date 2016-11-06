#! /usr/bin/env python3
'''
Very basic machine learning
link: https://www.youtube.com/watch?v=cKxRvEZd3Mw
'''

from sklearn import tree

'''
features
0 = bumpy,
1 = smooth

labels
0 = apples,
1 = oranges
'''

lookup = ["apples", "oranges"]
features =  [
                [140, 1],
                [130, 1],
                [130, 0],
                [130, 0]
            ]
labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(lookup[
    int(clf.predict([[160, 0]]))
    ])

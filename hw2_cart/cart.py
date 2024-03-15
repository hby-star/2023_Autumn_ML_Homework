# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:50:17 2023

@author: 21714
"""
import numpy as np
import random
import csv
from sklearn import tree
from graphviz import Source


"""
read csv
select age and buy_qty as X
select GMB as Y
"""
p = r'eBay_business_case_v3.0.csv'
with open(p,encoding = 'utf-8') as f:
    X = np.loadtxt(f,str,delimiter = ",", skiprows = 1,usecols = (5,7))
with open(p,encoding = 'utf-8') as f:
    Y = np.loadtxt(f,str,delimiter = ",", skiprows = 1,usecols = (8))
"""
np.random.seed(42)
X=np.random.randint(10, size=(100, 4))
Y=np.random.randint(2, size=100)
a=np.column_stack((Y,X))
"""                               
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=3)
clf = clf.fit(X, Y)
graph = Source(tree.export_graphviz(clf, out_file=None))
graph.format = 'png'
graph.render('cart_tree',view=True)


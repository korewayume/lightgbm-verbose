# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_iris

iris = load_iris()
indexes = np.where(iris.target != 2)
iris.target[iris.target != 1] = 0
X = pd.DataFrame(iris.data[indexes], columns=iris.feature_names)
Y = pd.DataFrame(iris.target[indexes], columns=['target'])
gbm = lgb.LGBMClassifier(num_leaves=3, max_depth=2, n_estimators=2, objective='binary', num_class=1)
gbm.fit(X, Y)

print("\nFeature:")
print(np.squeeze(X[:1]))
print("\nTarget:")
print(np.array(Y[:1]).squeeze().tolist())
print("\nPredict")
print(np.squeeze(gbm.predict_proba(X[:1])))
print("\nPredict Leaf Index")
print(np.squeeze(gbm.predict_proba(X[:1], pred_leaf=True)))


def flat_leaf(node):
    leaves = []
    if 'leaf_index' in node:
        leaves.append(node)
    if 'left_child' in node:
        leaves.extend(flat_leaf(node['left_child']))
    if 'right_child' in node:
        leaves.extend(flat_leaf(node['right_child']))
    return leaves


for tree_index, tree in enumerate(gbm.booster_.dump_model()['tree_info']):
    print('\ntree_index:', tree_index)
    for leaf in flat_leaf(tree['tree_structure']):
        print(leaf)

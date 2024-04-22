# Decision Tree

## Tasks:

### 0. Depth of a decision tree:
All the nodes of a decision tree have their `depth` attribute. The depth of the root is `0` , while the children of a node at depth `k` have a depth of `k+1`. We want to find the maximum of the depths of the nodes (including the leaves) in a decision tree. In order to do so, we added a method `def depth(self):` in the `Decision_Treeclass`, a method `def max_depth_below(self):` in the `Leaf` class.

Task: Update the class `Node` by adding the method `def max_depth_below(self):`.
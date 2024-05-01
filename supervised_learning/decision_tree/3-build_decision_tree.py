#!/usr/bin/env python3
""" This module handles the definition and behavior of a decision_tree,
    contains:

    classes:
        1. Node: Nodes of the tree.
        2. Leaf: Leaves of the tree.
        3. Decision_Tree: Decision tree.

    requires:
        - numpy.
"""

import numpy as np


class Node:
    """ Defines a decision tree node """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """ Initializes a node
            - feature: int, feature of the dataset that is used
                       to split the decision tree.
            - threshold: int, threshold.
            - left_child: node, left child.
            - right_child: node, right child.
            - is_root: boolean, determines whether the node is the tree root.
            - depth: int, tree depth.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Computes the maximum depth of the subtree below this node """
        if self.is_leaf:
            return self.depth

        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """ Counts the number of nodes
            - only_leaves: determines whether to count only leaves
        """
        def cleaves(node):
            """ Counts leaves recursively
                - node: node to count from
            """
            if node is None:
                return 0
            if node.left_child is None and node.right_child is None:
                return 1
            return cleaves(node.left_child) + cleaves(node.right_child)

        def cnodes(node):
            """ Counts nodes recursively
                - node: node to count from
            """
            if node is None:
                return 0
            return 1 + cnodes(node.left_child) + cnodes(node.right_child)

        return cleaves(self) if only_leaves else cnodes(self)

    def left_child_add_prefix(self, text):
        """ Adds a prefix to left children
                - text: text to add prefix to
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"

        return new_text

    def right_child_add_prefix(self, text):
        """ Adds a prefix to right children
                - text: text to add prefix to
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text.rstrip()

    def get_leaves_below(self):
        """ Gets all the leaves found deeper """
        leaves = []
        self._get_leaves_recursive(leaves)
        return leaves

    def _get_leaves_recursive(self, leaves):
        """ Helper function for recursive leaf retrieval """
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                self.left_child._get_leaves_recursive(leaves)
            if self.right_child:
                self.right_child._get_leaves_recursive(leaves)

    def __str__(self):
        """ String representation for the class Node """
        dtree = ""

        if self.is_root:
            dtree += (
                f"root [feature={self.feature}, threshold={self.threshold}]\n"
            )
        elif self.is_leaf:
            dtree += f"-> leaf [value={self.value}]"
        else:
            dtree += (
                f"-> node ["
                f"feature={self.feature}, "
                f"threshold={self.threshold}]\n"
            )

        if self.left_child:
            dtree += self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            dtree += self.right_child_add_prefix(str(self.right_child))

        return dtree


class Leaf(Node):
    """ Defines a tree leaf """
    def __init__(self, value, depth=None):
        """ Initializes a Leaf
            - value: int, value it contains.
            - depth: int, depth of it.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Maximum depth you can go """
        return self.depth

    def get_leaves_below(self):
        """ Gets all leaves found deeper """
        return [self]

    def __str__(self):
        """ String representation for the class Leaf """
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """ Defines a decision tree """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """ Initializes a Decision Tree
            - max_depth:
            - min_pop:
            - seed:
            - split_criterion:
            - root:
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ Calculates its depth """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Counts the nodes in a decision tree
            - only_leaves: determines whether count only leaves
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """ Gets all leaves found deeper """
        return self.root.get_leaves_below()

    def __str__(self):
        """ String representation for the class Decision Tree """
        return self.root.__str__() + "\n"

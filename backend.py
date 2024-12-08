import joblib
from sklearn.tree import _tree

# Load the decision tree model
filename = "decision_tree_model.pkl"  # Replace with your model's file name
decision_tree = joblib.load(filename)

# Function to traverse the decision tree
def traverse_tree(tree, node_id=0):
    """
    Traverse a decision tree starting from the given node.

    Args:
        tree: sklearn tree._tree.Tree object
        node_id: ID of the current node (default is root node 0)
    """
    if tree.feature[node_id] != _tree.TREE_UNDEFINED:
        # Internal node
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        print(f"Node {node_id}: If feature[{feature}] <= {threshold:.2f}")
        
        # Recursively traverse the left and right children
        traverse_tree(tree, tree.children_left[node_id])
        print(f"Node {node_id}: Else")
        traverse_tree(tree, tree.children_right[node_id])
    else:
        # Leaf node
        value = tree.value[node_id]
        print(f"Leaf {node_id}: Predict {value}")

# Extract the tree structure
tree_structure = decision_tree.tree_

# Traverse the decision tree
traverse_tree(tree_structure)

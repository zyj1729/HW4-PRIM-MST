import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    Tests the functionality of the `construct_mst` method within the `Graph` class to ensure it correctly constructs
    a Minimum Spanning Tree (MST) from a given adjacency matrix of an undirected graph.

    The test performs the following checks:
    1. Verifies that the total number of edges in the MST is exactly one less than the number of vertices, 
       which is a characteristic property of an MST in a connected graph.
    2. Ensures that the MST is fully connected, meaning there should be no isolated vertices.

    The test uses a predefined adjacency matrix representing a simple graph with three vertices and three edges.
    After constructing the MST using the `construct_mst` method, it examines the resulting MST to confirm
    it meets the expected properties of an MST.

    The function does not return anything but uses assertions to validate the MST's properties.
    """

    # Define a simple adjacency matrix for testing
    adj_matrix = np.array([
        [0, 1, 3],  # Edge weights from the first vertex to others
        [1, 0, 1],  # Edge weights from the second vertex to others
        [3, 1, 0]   # Edge weights from the third vertex to others
    ])

    # Initialize the Graph with the test adjacency matrix and construct the MST
    g = Graph(adj_matrix)
    g.construct_mst()
    mst = g.mst

    # Initialize a variable to count the total number of edges in the MST
    total = 0

    # Flag to check if the MST is fully connected
    connect = True

    # Loop through each row of the MST adjacency matrix to count edges and check connectivity
    for i in range(mst.shape[0]):
        # Count the number of non-zero elements in the upper triangle including the diagonal
        # to avoid counting edges twice in this undirected graph representation
        total += len(np.nonzero(mst[i][:i+1])[0])

        # Check if the current vertex is connected to any other vertex
        add = len(np.nonzero(mst[i])[0])
        if not add:
            connect = False  # If a vertex is not connected, mark the MST as not fully connected

    # Assert that the total number of edges in the MST is exactly one less than the number of vertices
    assert total == mst.shape[0] - 1, 'Proposed MST has incorrect number of edges'

    # Assert that the MST is fully connected with no isolated vertices
    assert connect, 'Proposed MST is not fully connected'

import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
            assert self.adj_mat.shape[0] > 1, "Input matrix dimension should be at least 2"
            assert self.adj_mat.shape[0] == self.adj_mat.shape[0], "Input matrix should have the same number of rows and cols"
        elif type(adjacency_mat) == np.ndarray:
            assert adjacency_mat.shape[0] > 1, "Input matrix dimension should be at least 2"
            assert adjacency_mat.shape[0] == adjacency_mat.shape[0], "Input matrix should have the same number of rows and cols"
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
        Constructs the Minimum Spanning Tree (MST) for the graph represented by the adjacency matrix `self.adj_mat`
        using Prim's algorithm. The MST is stored as an adjacency matrix in `self.mst`.

        Prim's algorithm starts with a single vertex and grows the MST by adding the cheapest edge from the graph
        that connects a vertex in the MST to a vertex outside the MST, until all vertices are included.

        Attributes:
        - self.adj_mat: The adjacency matrix of the input graph.
        - self.mst: The adjacency matrix representing the MST of the input graph.
        """

        # Initialize the MST adjacency matrix with zeros
        self.mst = np.zeros(self.adj_mat.shape)

        # Randomly select an initial vertex
        ini = np.random.randint(self.adj_mat.shape[0])

        # S: Set of vertices included in the MST
        S = [ini]

        # T: Priority queue (min-heap) of edges in the format (weight, source, target)
        T = []
        heapq.heapify(T)

        # Add all edges from the initial vertex to the priority queue
        for i in range(len(self.adj_mat[ini])):
            if i not in S and self.adj_mat[ini][i] != 0:
                heapq.heappush(T, (self.adj_mat[ini][i], ini, i))

        # Loop until all vertices are included in S
        while len(S) < self.adj_mat.shape[0]:
            while True:
                # Pop the cheapest edge from T
                edge = heapq.heappop(T)
                # If the target of the edge is not in S, it's a valid edge to add to the MST
                if edge[2] not in S:
                    # Add the target vertex to S
                    S.append(edge[2])
                    # Update the MST adjacency matrix with the new edge
                    self.mst[edge[1]][edge[2]] = edge[0]
                    self.mst[edge[2]][edge[1]] = edge[0]
                    # Add all new edges from the newly added vertex to T
                    for i in range(len(self.adj_mat[edge[2]])):
                        if i not in S and self.adj_mat[edge[2]][i] != 0:
                            heapq.heappush(T, (self.adj_mat[edge[2]][i], edge[2], i))
                    break  # Break the inner loop to start looking for the next cheapest edge
        # No explicit return needed as the MST is stored in self.mst

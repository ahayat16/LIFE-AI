# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:59:30 2020
@author: Nathaniel
"""


import numpy as np
import time
import math


from scipy.sparse import csr_matrix
from scipy.sparse import diags
from networkx import nx


def findpaths(Gtest):
    # find nodes without paths to the excretion. For each node without a path
    # to excretion, add it to the nodeswithout paths list. Return list when
    # finished.
    nodeswithoutpaths = []
    for i in range(Gtest.number_of_nodes()):
        if not nx.has_path(Gtest, i, Gtest.number_of_nodes() - 1):
            nodeswithoutpaths.append(i)
    return nodeswithoutpaths


# verbose=False disables printed messages
def find_eq(Gtest, phi, excrete, verbose=True):
    # Find the equilibrium. First Create the Laplacian from the adjacency and
    # diagonal matrix. Add the excretions to form a Grounded Laplacian.
    # The LG should be invertiblex, the formula for equilibrium is solving the equation
    # transpose(LG(x)) = phi,  thus x = inverse(LG.T) *phi where phi is intakes
    A = nx.adjacency_matrix(Gtest)
    Diagonal = diags(np.ndarray.tolist(np.reshape(A.sum(axis=1), (1, A.shape[0])))[0])

    L = Diagonal - A
    LG = L + diags(excrete)
    if (
        np.linalg.matrix_rank(csr_matrix.todense(LG.T))
        != np.shape(csr_matrix.todense(LG.T))[0]
    ):
        if verbose:
            print("Grounded Laplacian not invertible")
        return None
    xmets = np.array(np.linalg.inv(csr_matrix.todense(LG.T)).dot(phi))[0]
    if verbose and not np.allclose(
        csr_matrix.todense(LG.T).dot(xmets) - phi, np.zeros_like(phi)
    ):
        print("EQnotfound")
    return xmets


# verbose=False disables printed messages
def test_eq(Adj, equil, verbose=True):
    # CONVERT THE GRAPH TO STOICHIOMETRIC MATRIX AND VERIFY THE EQUILIBRIUM WORKS.
    assert type(Adj) == csr_matrix or type(Adj) == np.ndarray

    if type(Adj) == csr_matrix:
        num_edges = csr_matrix.getnnz(Adj)
    else:
        num_edges = int(np.count_nonzero(Adj))

    num_nodes = Adj.shape[0]
    S = np.zeros((num_nodes, num_edges))
    list_edges = np.nonzero(Adj)
    equil_with_intake = np.append(1, equil)

    for i in range(num_edges):
        S[[list_edges[0][i], list_edges[1][i]], i] = (
            np.array((-1, 1)) * equil_with_intake[list_edges[0][i]] * Adj[list_edges][i]
        )

    Sgrounded = S[1:-1, :]

    # ROWS SHOULD SUM TO ZERO AT EQUILIBRIUIM
    if verbose and not np.allclose(
        np.sum(Sgrounded, axis=1), np.zeros_like(np.sum(Sgrounded, axis=1))
    ):
        print("EQ_Fail")
    return np.allclose(
        np.sum(Sgrounded, axis=1), np.zeros_like(np.sum(Sgrounded, axis=1))
    )


def add_edges_random(Gtest, weighted: bool):
    # Randomly add edges to the graph.
    nonedges = list(nx.non_edges(Gtest))

    chosen_nonedge = np.random.randint(len(nonedges))

    # force to have no edges from excretion or to intake
    while (
        nonedges[chosen_nonedge][0] == Gtest.number_of_nodes() - 1
        or nonedges[chosen_nonedge][1] == 0
    ):
        chosen_nonedge = np.random.randint(len(nonedges))
    if weighted is True:
        chosen_weight = np.random.randint(1, 100)
        Gtest.add_edge(
            nonedges[chosen_nonedge][0],
            nonedges[chosen_nonedge][1],
            weight=chosen_weight,
        )
    else:
        Gtest.add_edge(nonedges[chosen_nonedge][0], nonedges[chosen_nonedge][1])

    return Gtest


def add_edges_targeted(Adj, non_eq_nodes, weighted: bool):
    newlist0 = np.random.choice(non_eq_nodes)
    newlist1 = newlist0

    # force not to add a loop
    while newlist1 == newlist0:
        newlist1 = np.random.choice(np.where(Adj[newlist0, 1:] == 0)[0]) + 1

    if weighted is True:
        # add random weight
        Adj[newlist0, newlist1] = np.random.randint(1, 100)
    else:
        Adj[newlist0, newlist1] = 1

    return Adj


def create_eq(Gtest, non_eq_nodes, weighted: bool, addrandom=False):
    newG = Gtest  # create a new graph

    # While there are nodes without paths, add edges
    while len(non_eq_nodes) > 0:
        # add edges to the new graph
        if addrandom:
            newG = add_edges_random(Gtest, weighted)
            Adj = np.squeeze(np.asarray(csr_matrix.todense(nx.adjacency_matrix(newG))))
            GroundedGraph = nx.convert_matrix.from_numpy_matrix(
                Adj[1:-1, 1:-1], create_using=nx.DiGraph
            )

        else:
            Adj = np.squeeze(np.asarray(csr_matrix.todense(nx.adjacency_matrix(newG))))
            Adj = add_edges_targeted(Adj, non_eq_nodes, weighted)
            newG = nx.convert_matrix.from_numpy_matrix(Adj, create_using=nx.DiGraph)
            GroundedGraph = nx.convert_matrix.from_numpy_matrix(
                Adj[1:-1, 1:-1], create_using=nx.DiGraph
            )
        # find paths and any non_eq_nodes
        non_eq_nodes = findpaths(newG)
    # find the equilibrium for the graph with added edges
    equilibrium = find_eq(GroundedGraph, Adj[0, 1:-1], Adj[1:-1, -1])
    return newG, equilibrium


def modified_watts_strogatz_graph(nr_nodes: int, nr_link: int, proba_rew: float):
    G = nx.watts_strogatz_graph(nr_nodes, nr_link, proba_rew)
    C = np.array(nx.convert_matrix.to_numpy_matrix(G))
    assert np.allclose(C, C.T)
    for i in range(len(C)):
        for j in range(i, len(C[i])):
            if C[i][j] != 0:
                if np.random.rand() > 0.5:
                    C[i][j] = 0
                else:
                    C[j][i] = 0
    GG = nx.convert_matrix.from_numpy_matrix(C, create_using=nx.DiGraph)
    return GG


def createGraphs(Quantity, NumNodes):
    # 3 lists to return
    noneqparamslist = []  # list of graphs with no equilibrium
    pureeqparamslist = []  # list of graphs with an equlibrium
    forcedeqparamslist = []  # list of graphs forced to have an equilibrium
    for i in range(Quantity):  # create Quantity number of graphs
        N = NumNodes  # Number of nodes
        m = round(N * (2 + np.random.rand() * 2))  # 2N-4N edges
        G = nx.gnp_random_graph(
            N, m / (N * (N - 1)), directed=True
        )  # Erdos random graph.

        C = np.array(
            nx.convert_matrix.to_numpy_matrix(G)
        )  # Convert graph to adjacency matrix
        D = np.append(C, np.zeros((1, C.shape[1])), axis=0)
        D = np.append(D, np.zeros((D.shape[0], 1)), axis=1)
        D[np.random.randint(len(D) - 1), -1] = 1

        D = np.append(np.zeros((1, D.shape[1])), D, axis=0)
        D = np.append(np.zeros((D.shape[0], 1)), D, axis=1)
        D[0, np.random.randint(1, len(D) - 1)] = 1

        # CREATE A NEW GRAPH USING ADJACENCY WITH INTAKE/EXCRETION
        GG = nx.convert_matrix.from_numpy_matrix(D, create_using=nx.DiGraph)

        # FIND NODES WITH PATH FROM INTAKE WITHOUT PATHS TO EXCRETION
        # PRESENCE OF THESE NODES WILL INDICATE NO EQUILIBRIUM
        non_eq_nodes = findpaths(GG)

        if not non_eq_nodes:  # If there is a natural equilibrium
            # FIND THE EQUILIBRIUM
            equilibrium = find_eq(G, D[0, 1:-1], D[1:-1, -1])

            # IF DESIRED THE EQUILIBRIUM CAN BE CHECKED
            if test_the_eq:
                test_eq(D, equilibrium)
            # ADD graph to pureeqparamslist
            pureeqparamslist.append(
                [GG.number_of_nodes(), GG.number_of_edges(), nx.edges(GG), equilibrium]
            )
        else:  # If there is not a natrual equilibrium
            equilibrium = math.nan  # set a value for equilibrium
            # IF THE NUMBER OF FORCED EQUILIBRIUM GRAPHS IS LESS THAN DESIRED,
            # COMPLETE THE GRAPH TO CREATE AN EQUILIBRIUM
            if len(forcedeqparamslist) / 2 < Quantity * (
                force_to_eq
            ):  # this should give around 30% of non_eq graphs get completed.
                # ADD THE ORIGINAL NON_EQ GRAPH PARAMS
                forcedeqparamslist.append(
                    [
                        GG.number_of_nodes(),
                        GG.number_of_edges(),
                        nx.edges(GG),
                        equilibrium,
                    ]
                )
                GG, equilibrium = create_eq(GG, non_eq_nodes, weighted, addrandom)
                # ADD THE NEW EDGE ADDED GRAPH WITH AN EQ
                forcedeqparamslist.append(
                    [
                        GG.number_of_nodes(),
                        GG.number_of_edges(),
                        nx.edges(GG),
                        equilibrium,
                    ]
                )
                if test_the_eq:
                    test_eq(nx.adjacency_matrix(GG), equilibrium)
            else:
                # if creating an equilibrium is not desired add it
                # to the noneqparamslist
                noneqparamslist.append(
                    [
                        GG.number_of_nodes(),
                        GG.number_of_edges(),
                        nx.edges(GG),
                        equilibrium,
                    ]
                )

    return pureeqparamslist, forcedeqparamslist, noneqparamslist


if __name__ == "__main__":
    # timing
    start = time.time()

    # This is an additional check to verify that a graph is at equilibrium
    # when it is stated to be.
    # If any graph fails it will print "EQ_fail"
    test_the_eq = False
    # If addrandom=True, then edges will be added randomly until equilibrium is reached.
    addrandom = False
    weighted = True
    # proportion of non_eq graphs to be completed. Should be between 0 and 1.
    force_to_eq = 0.0

    # createGraphs(Quantity,NumNodes) creates Quantity number of graphs
    # with NumNodes nodes.
    # returns a list of 3 lists.
    # each of these three lists has [number of nodes, number of edges, list of edges,
    # equilibrium (if one exists)]
    # List 0 is the graphs with equilibrium
    # List 1 is the graphs with created equilibrium (both original and modified graph)
    # List 2 is the graphs with no equilibrium
    Allgraphs = createGraphs(100, 100)

    # end timing
    end = time.time()

    # Statistics about the graphs
    print(end - start, "Seconds for completion")
    a = np.array(Allgraphs[0])
    print(len(a), "Number of random graphs with equilibrium")
    print(np.mean(a[:, 1]), "Average number of edges for graphs with equilibrium")
    aa = np.array(Allgraphs[1])
    print(len(aa) / 2, "Number of graphs with created equilibrium")
    if len(aa) > 0:
        print(
            np.mean(aa[1::2, 1]),
            "Average number of edges after equilibrium was created",
        )
        print(
            np.mean(aa[::2, 1]),
            "Average number of edges before equilibrium was created",
        )
    aaa = np.array(Allgraphs[2])
    print(len(aaa), "Number of graphs with no equilibrium")
    print(np.mean(aaa[:, 1]), "Average number of edges when no equilibrium was found")

    fraction = []
    proba_link = 3
    for i in range(100):
        nr_nodes = np.random.randint(1, 25)
        G = nx.scale_free_graph(
            nr_nodes,
            alpha=0.99 / proba_link,
            beta=(proba_link - 1) / proba_link,
            gamma=0.01 / proba_link,
        )
        C = np.array(nx.convert_matrix.to_numpy_matrix(G))
        fraction.append([nr_nodes, np.count_nonzero(C) / nr_nodes])
    frac = sorted(fraction, key=lambda frac: frac[0])

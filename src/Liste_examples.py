import json
import networkx as nx
import copy
from src.envs.graph import GraphEnvironment
from scipy.sparse import csr_matrix

# from src.envs.Open_files import graph_list_modified
# cd users/hayat/downloads/LIFE-Master-master_10
from src.envs.Open_files import graph_as_list
from src.envs.Open_files import Networks
import GraphCreationLIFE as gcl
import numpy as np


Example_1 = {}
Example_1_nodes = [
    [1, ["PEP"], [2, 14]],
    [2, ["Pyruvate"], [1, 3, 4]],
    [3, ["Acetyl-CoA"], [7, 13]],
    [4, ["CO2"], []],
    [5, ["Acetate"], [3]],
    [6, ["HCO3-"], [14]],
    [7, ["Citrate"], [8]],
    [8, ["Isocitrate"], [7, 9, 4, 15, 11]],
    [9, ["2-OG"], [10, 4]],
    [10, ["SSA"], [11]],
    [11, ["Succinate"], [10, 12, 16]],
    [12, ["Fumarate"], [13]],
    [13, ["Malate"], [12, 4, 2, 14]],
    [14, ["Oxaloacetate"], [13, 7]],
    [15, ["Glyoxylate"], [13]],
    [16, ["Succinil-CoA"], [11]],
]
for c in Example_1_nodes:
    c[0] = c[0] - 1
    c2bis = []
    for d in c[2]:
        c2bis.append(d - 1)
    c[2] = c2bis

Example_2 = {}
# RCT

Example_2_nodes = [
    [0, ["v0"], [1, 2, 3]],
    [1, ["v1"], [4]],
    [2, ["v2"], [4]],
    [3, ["v3"], [4]],
    [4, ["v4"], [5, 6]],
    [5, ["v5"], [6]],
    [6, ["v6"], [7]],
    [7, ["v7"], []],
]


class params:
    def __init__(self):
        self.max_len = 640
        self.min_nodes = 32
        self.max_nodes = 64
        self.min_edges = 2
        self.max_edges = 4
        self.erdos = True
        self.redeem_prob = 0.4
        self.add_random = False
        self.predict_eq = False
        self.float_precision = 4
        self.float_tolerance = 0.1
        self.weighted = False
        self.generator = "erdos"
        self.more_tolerance = "1"
        self.multiple_intake_excr = True
        self.nb_max_intake = 100
        self.nb_max_excr = 100
        self.nb_min_intake = 1
        self.nb_min_excr = 1
        self.tokenized_weights = True
        self.weighted = True
        self.proba_rewriting = None
        self.rng = None


par = params()
env = GraphEnvironment(par)


def From_example_2(Network: list):
    non_intake = []
    non_excr = []
    g = Network
    g1 = copy.deepcopy(g)
    for edge in g1["edges"]:
        assert edge[0] in g1["nodes"] and edge[1] in g1["nodes"], (edge[0], edge[1])
        non_intake.append(edge[1])
        non_excr.append(edge[0])
    non_intake = set(non_intake)
    non_intake = list(non_intake)
    non_excr = set(non_excr)
    non_excr = list(non_excr)
    intake = [node for node in g["nodes"] if node not in non_intake]
    excr = [node for node in g["nodes"] if node not in non_excr]
    for node in intake:
        assert node not in excr, (node, Network["edges"])
    for node in excr:
        assert node not in intake, (node, Network["edges"])
    for i in range(len(g1["edges"])):
        if g1["edges"][i][0] in intake:
            g1["edges"][i][0] = "intake"
        if g1["edges"][i][1] in excr:
            g1["edges"][i][1] = "excr"
    for node in Network["nodes"]:
        if node in intake:
            assert node not in excr, (node, Network["edges"])
            g1["nodes"].remove(node)
        if node in excr:
            g1["nodes"].remove(node)

    assert len(g1["nodes"]) == len(Network["nodes"]) - len(intake) - len(excr)

    g1["nodes"].append("excr")
    g1["nodes"].insert(0, "intake")

    assert len(g1["nodes"]) == len(Network["nodes"]) - (len(intake) - 1) - (
        len(excr) - 1
    )

    # Create dictionnary
    dico_node = {}
    g2 = []
    G3 = nx.DiGraph()
    for i in range(len(g1["nodes"])):
        dico_node.update({g1["nodes"][i]: i})
        G3.add_node(i)
        if i == len(g1["nodes"]) - 1:
            assert g1["nodes"][i] == "excr"
        elif i == 0:
            assert g1["nodes"][i] == "intake"
    for i in range(len(g1["edges"])):
        source = dico_node[g1["edges"][i][0]]
        target = dico_node[g1["edges"][i][1]]
        g2.append([source, target])

    G3.add_edges_from(g2)

    return G3


def From_example_graph(Example_nodes: list):
    """
    To be used only for the examples manually tabulated above
    """
    non_intake = []
    Ex_nodes = copy.deepcopy(Example_nodes)
    for c in Ex_nodes:
        non_intake.extend([node for node in c[2]])
        print([node for node in c[2]])
    non_intake_set = set(non_intake)

    intake_nodes = [c[0] for c in Ex_nodes]
    excretion_nodes = [c[0] for c in Ex_nodes if c[2] == []]
    for c in non_intake_set:
        if c in intake_nodes:
            intake_nodes.remove(c)

    New_nodes = []
    for c in Ex_nodes:
        New_nodes.append(c)
    permutation = {}
    for j in range(len(New_nodes)):
        permutation.update({j: j})
    for i in range(len(intake_nodes)):
        for j in range(intake_nodes[i]):
            permutation.update({j: permutation[j] + 1})
    for i in range(len(excretion_nodes)):
        for j in range(excretion_nodes[i], len(New_nodes)):
            permutation.update({j: permutation[j] - 1})
    print("#Before intake")
    print(permutation)

    for i in range(len(intake_nodes)):
        print(intake_nodes[i])
        permutation.update({intake_nodes[i]: i})

    print("#After intake")
    print(permutation)
    for i in range(len(excretion_nodes)):
        permutation.update({excretion_nodes[i]: len(New_nodes) - 1 - i})
    print("#After excre")
    print(permutation)
    print("===")
    print("Examples_nodes")
    print(Example_nodes)
    print("Ex_nodes")
    print(Ex_nodes)
    print("New_nodes")
    print(New_nodes)
    for c in New_nodes:
        c[0] = permutation[c[0]]
        new_c2 = []
        for d in c[2]:
            new_c2.append(permutation[d])
        c[2] = new_c2
    print("Examples_nodes")
    print(Example_nodes)
    print("Ex_nodes")
    print(Ex_nodes)
    print("New_nodes")
    print(New_nodes)

    assert len(New_nodes) == len(Example_nodes)

    for c in New_nodes:
        new_c2 = []
        for node in c[2]:
            if node >= (len(New_nodes) - len(excretion_nodes)):
                new_c2.append(len(New_nodes) - len(excretion_nodes))
            else:
                new_c2.append(node)
        c[2] = new_c2
        c[2] = set(c[2])
        c[2] = list(c[2])
    New_nodes = New_nodes[: (len(New_nodes) - len(excretion_nodes) + 1)]

    target_intake = []
    for i in range(len(intake_nodes)):
        target_intake.extend(New_nodes[i][2])
    target_intake = set(target_intake)
    target_intake = list(target_intake)
    if len(intake_nodes) > 0:
        New_nodes = New_nodes[len(intake_nodes) - 1 :]
        New_nodes[1][2] = target_intake

    G3 = nx.DiGraph()
    for c in New_nodes:
        G3.add_node(c[0])
        if len(c[2]) > 0:
            for node in c[2]:
                G3.add_edge(c[0], node)
    return G3


def From_example_to_input(
    Example_nodes: list, weighted: bool, tokenized_weights: bool, predict_eq: bool
):
    G = From_example_2(Example_nodes)
    non_eq_nodes = gcl.findpaths(G)

    D = np.array(nx.convert_matrix.to_numpy_matrix(G))

    GroundedGraph = nx.convert_matrix.from_numpy_matrix(
        D[1:-1, 1:-1], create_using=nx.DiGraph
    )
    if len(GroundedGraph.nodes) == 0:
        print("lol")
        return None

    if not non_eq_nodes and predict_eq:
        equilibrium = gcl.find_eq(GroundedGraph, D[0, 1:-1], D[1:-1, -1], False)

    result = 0 if non_eq_nodes else 1

    nr_nodes = len(G.nodes)

    x = []
    x.append(f"N{nr_nodes}")
    for n in G.edges(data=True):
        x.append(f"N{n[0]}")
        x.append(f"N{n[1]}")
        if weighted:
            w = int(n[2]["weight"])
            if tokenized_weights:
                x.append(f"N{w}")
            else:
                x.extend(env.write_int(w))

    y = [f"N{result}"]
    if result == 1 and predict_eq:
        for val in equilibrium:
            y.append(env.separator)
            y.extend(env.write_float(val))

    return x, y


if __name__ == "__main__":

    data_input = []
    data_graphs = []
    ratios = []
    len_graphs = []
    i = 0
    for n in Networks:
        i += 1
        print(i)
        ratio = len(n["edges"]) / len(n["nodes"])
        ratios.append(ratio)
        len_graphs.append(len(n["nodes"]))
        data_input.append(From_example_to_input(n, False, False, True))
        GG = From_example_2(n)
        list_edges = list(GG.edges)
        list_nodes = list(GG.nodes)
        data_graphs.append({"edges": list_edges, "nodes": list_nodes})

    with open("data_quanti.json", "w") as f:
        json.dump({"data_input": data_input, "data_graphs": data_graphs}, f)


import os
import json
import numpy as np
import networkx as nx
import jgf.igraph as jig
import xml.etree.ElementTree as ET


####### KGML Data ########
file_path = "src/Data/KEGG"
Networks = []
nr_file = 0
for filename in os.listdir(file_path):
    nr_file += 1
    # try:
    nodes = []
    edges = []
    tree = ET.parse(os.path.join(file_path, filename))
    root = tree.getroot()
    empty_graph = 1  # By defaut we do not record the graph if it has no edge and nodes
    cancel_graph = 0  # We record the graph if not empty
    for child in root:
        if child.tag == "reaction":
            empty_graph = 0  # Graph is not empty
            source = []
            target = []
            for c in child:
                if c.tag == "substrate":
                    source.append(c.attrib["name"])
                if c.tag == "product":
                    target.append(c.attrib["name"])
            if len(target) > 1:
                cancel_graph = 1  # We do not record the graph either if a reaction has several product (complicated hyperedge)
            else:
                assert len(target) == 1
            nodes.extend(source)
            nodes.extend(target)
            for i in range(len(source)):
                if [source[i], target[0]] not in edges:
                    edges.append([source[i], target[0]])
            if child.attrib["type"] == "reversible":
                for i in range(len(source)):
                    if [target[0], source[i]] not in edges:
                        edges.append([target[0], source[i]])
            nodes = set(nodes)
            nodes = list(nodes)
    if cancel_graph == 0 and empty_graph == 0:  # Check if we record the graph
        Networks.append({"nodes": nodes, "edges": edges})
        print(root.attrib["title"])

graph_as_list = []
for g in Networks:
    dico_nodes = {}
    dico_edges = {}
    node_list = []
    for i in range(len(g["nodes"])):
        dico_nodes.update({g["nodes"][i]: i})
        node_list.append([i, g["nodes"][i], []])
    for edge in g["edges"]:
        assert node_list[dico_nodes[edge[0]]][0] == dico_nodes[edge[0]]
        node_list[dico_nodes[edge[0]]][2].append(dico_nodes[edge[1]])
    graph_as_list.append(node_list)


####### KEGG Data 2 ########
file_path = "src/Data/KEGG"
Networks_2 = []
nr_file = 0
for filename in os.listdir(file_path):
    nr_file += 1
    nodes = []
    edges = []
    tree = ET.parse(os.path.join(file_path, filename))
    root = tree.getroot()
    empty_graph = 1  # By defaut we do not record the graph if it has no edge and nodes
    cancel_graph = 0  # We record the graph if not empty
    for child in root:
        if child.tag == "reaction":
            cancel_graph = 1
        if child.tag == "relation" and len(child) > 1:
            if child[0].attrib["name"] in ["activation", "expression", "dissociation"]:
                empty_graph = 0  # Graph is not empty
                source = child.attrib["entry1"]
                target = child.attrib["entry2"]
                nodes.append(source)
                nodes.append(target)
                edges.append([source, target])

            elif child[0].attrib["name"] in ["inhibition", "repression"]:
                # print("Not implemented yet")
                continue
            elif child[0].attrib["name"] == "binding/association":
                source = child.attrib["entry1"]
                target = child.attrib["entry2"]
                nodes.append(source)
                nodes.append(target)
                edges.append([source, target])
                edges.append([target, source])
            else:
                name = child[0].attrib["name"]
                print(f"Name not found {name}")
                cancel_graph = 1
    if cancel_graph == 0 and empty_graph == 0:  # Check if we record the graph
        nodes = set(nodes)
        nodes = list(nodes)
        Networks_2.append({"nodes": nodes, "edges": edges})
        print(root.attrib["title"])

graph_as_list_2 = []
j = 0
for g in Networks_2:
    print(j)
    dico_nodes = {}
    dico_edges = {}
    node_list = []
    for i in range(len(g["nodes"])):
        dico_nodes.update({g["nodes"][i]: i})
        node_list.append([i, g["nodes"][i], []])
    for edge in g["edges"]:
        assert node_list[dico_nodes[edge[0]]][0] == dico_nodes[edge[0]]
        node_list[dico_nodes[edge[0]]][2].append(dico_nodes[edge[1]])
    graph_as_list_2.append(node_list)
    j += 1

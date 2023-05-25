import torch
import networkx as nx

def get_ER_random_contact(n, avgDegree, device="cpu"):
    graph=nx.dense_gnm_random_graph(n, n*avgDegree)
    graph=nx.to_numpy_array(graph)
    contact = torch.FloatTensor(graph).to(device)
    return contact

def get_WS_random_contact(n, k, p, device="cpu"):
    #number of nodes
    #each node is joined with its k nearest neighbors
    #probability of rewiring each edge
    graph=nx.watts_strogatz_graph(n, k, p)
    graph=nx.to_numpy_array(graph)
    contact = torch.FloatTensor(graph).to(device)
    return contact

def get_BA_random_contact(n, m, device="cpu"):
    #Number of nodes
    #Number of edges to attach from a new node to existing nodes
    graph=nx.barabasi_albert_graph(n, m)
    graph=nx.to_numpy_array(graph)
    contact = torch.FloatTensor(graph).to(device)
    return contact
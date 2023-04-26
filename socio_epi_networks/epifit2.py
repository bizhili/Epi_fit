import torch
import pandas as pd
import random
from epifit.network_generation import generate_random_network
from epifit.graphing_utils import *
from matplotlib import pyplot as plt
import networkx as nx

device = "cuda" if torch.cuda.is_available() else "cpu"

def population(n:int, device= "cpu"):
  initial_population = { #a columnar DB somewhat reminds the Pandas DataFrame 
      "S":torch.ones(n).to(device),
      "E":torch.zeros(n).to(device),
      "I":torch.zeros(n).to(device),
      "R":torch.zeros(n).to(device),
  }
  initial_population["S"][0]=0 #first one infective
  initial_population["I"][0]=1 #first one infective
  return initial_population

def get_ER_random_contact(n, avgDegree):
    graph=nx.dense_gnm_random_graph(n, n*avgDegree)
    graph=nx.to_numpy_array(graph)
    contact = torch.FloatTensor(graph).to(device)
    return contact

_ps= torch.tensor([ [0, 0.2, 0, 0],
                    [0, 0, 0.2, 0],
                    [0, 0, 0, 0.2],
                    [0.2, 0, 0, 0]], device=device)

class EPI_dense(torch.nn.Module):
    def __init__(self, ISNet, psMatrix, population, device):
        super(EPI_dense, self).__init__()
        self._n= ISNet.size()[0]
        self._IS= ISNet
        self._EE= torch.eye(self._n).to(device)
        self._II= self._EE
        self._RR= self._EE
        self._state= torch.stack((population["S"], population["E"], population["I"], population["R"]))
        self._L= torch.zeros_like(self._state, device=device)
        self._psMatrix= psMatrix
        self._P= self._state
    
    def reset_population(self, population):
        self._state= torch.stack((population["S"], population["E"], population["I"], population["R"]))
        self._P= self._state

    def forward(self):
        self._L[0]= self._state[0]*torch.matmul(self._IS, self._state[2])
        self._L[1]= torch.matmul(self._state[1], self._EE)
        self._L[2]= torch.matmul(self._state[2], self._II)
        self._L[3]= torch.matmul(self._state[3], self._RR)

        self._P= 1- torch.exp(torch.matmul(torch.log(1 - self._psMatrix.T), self._L))
        _stable_prob= 1- torch.sum(self._P, 0)
        self._P= self._P+ self._state*_stable_prob
        self._state= self.rample_uniform_matrix(self._P)

        return self._state, self._P

    #sample nxm pobability matrix, of 0 dimension, which contains n choise for a random variable
    def rample_uniform_matrix(self, P):  
        state= torch.zeros_like(P, device=device)
        U= torch.rand(self._n).to(device)
        for i in range(P.size()[0]):
            U= U- P[i]
            state[i]= U<0
            U= U+state[i]
        return state
    
    def get_population_number(self):

        return torch.sum(self._state, 1)
    
    def get_population(self):
        return { #a columnar DB somewhat reminds the Pandas DataFrame 
                "S": self._state[0],
                "E": self._state[1],
                "I": self._state[2],
                "R": self._state[3],
                }
    def get_expectation(self):
        return torch.sum(self._P, 1)
    
    def get_probability_transition_matrix(self):
        
        return self._psMatrix

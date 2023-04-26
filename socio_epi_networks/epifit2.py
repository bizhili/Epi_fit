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


class EPI_dense(torch.nn.Module):
    def __init__(self, ISNet, psMatrix, population, device, train=False):
        super(EPI_dense, self).__init__()
        self._n= ISNet.size()[0]
        self._IS= ISNet
        self._EE= torch.eye(self._n).to(device)
        self._II= self._EE
        self._RR= self._EE
        self._state= torch.stack((population["S"], population["E"], population["I"], population["R"]))
        self._psMatrix= psMatrix
        self._train= train
        if train==True:
            self._psMatrix=torch.nn.Parameter(psMatrix)
        self._P= self._state
    
    def reset_population(self, population):
        self._state= torch.stack((population["S"], population["E"], population["I"], population["R"]))
        self._P= self._state

    def forward(self):
        stateGradient= torch.clone(self._state)
        L= torch.zeros_like(self._state, device=device)
        L[0]= stateGradient[0]*torch.matmul(self._IS, stateGradient[2])
        L[1]= torch.matmul(stateGradient[1], self._EE)
        L[2]= torch.matmul(stateGradient[2], self._II)
        L[3]= torch.matmul(stateGradient[3], self._RR)
        if self._train==False:
            prob= 1- torch.exp(torch.matmul(torch.log(1 - self._psMatrix.T), L))
        else:
            psMatrix= torch.sigmoid(self._psMatrix)
            prob= 1- torch.exp(torch.matmul(torch.log(1 - psMatrix.T), L))
        _stable_prob= 1- torch.sum(prob, 0)
        prob= prob+ stateGradient*_stable_prob

        with torch.no_grad():
            self._state= self.rample_uniform_matrix(prob)
            self._P= prob

        return torch.sum(self._state, 1), torch.sum(prob, 1)

    #sample nxm pobability matrix, of 0 dimension, which contains n choise for a random variable
    def rample_uniform_matrix(self, P):  
        state= torch.zeros_like(P, device=device)
        U= torch.rand(self._n).to(device)
        for i in range(P.size()[0]):
            U= U- P[i]
            state[i]= U<0
            U= U+state[i]
        return state
    
    def get_population(self):
        return { #a columnar DB somewhat reminds the Pandas DataFrame 
                "S": self._state[0],
                "E": self._state[1],
                "I": self._state[2],
                "R": self._state[3],
                }
    def get_population(self):

        return self._state
    
    def get_probability(self):
        return self._P
    
    def get_probability_transition_matrix(self):
        
        return self._psMatrix
    
    def get_psMatrix(self):

        return torch.sigmoid(self._psMatrix)


if __name__=="__main__":
    n= 1000
    avgDegree= 5
    timeHorizon= 20
    contact=get_ER_random_contact(n, avgDegree)
    contact = contact.requires_grad_(True)
    realData =None

    with torch.no_grad():
        ps= torch.tensor([[0, 0.2, 0, 0],
                        [0, 0, 0.99, 0],
                        [0, 0, 0, 0.0],
                        [0.0, 0, 0, 0]], device=device)
        population= population(n, device)
        model= EPI_dense(contact, ps, population, device)
    pass

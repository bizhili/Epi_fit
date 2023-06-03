import sys
sys.path.append('../')

import graph.random_graph as random_graph
import util.create_population as create_population

import torch
import pandas as pd
import random
from matplotlib import pyplot as plt
import networkx as nx

device = "cuda" if torch.cuda.is_available() else "cpu"

class pure_EPI_dense(torch.nn.Module):
    def __init__(self, ISNet, psMatrix, device, train=False, cc=None, recursive=False, sampleAsState=True):
        super(pure_EPI_dense, self).__init__()
        self._n= ISNet.size()[0]
        self._IS= ISNet
        self._EE= torch.eye(self._n).to(device)
        self._II= self._EE
        self._RR= self._EE
        self._psMatrix= psMatrix
        self._train= train
        if train==True:
            self._psMatrix=torch.nn.Parameter(psMatrix)
        self._P= None
        self._forceCc=torch.tensor([[0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0]], device=device) 
        self._cc= cc
        self._recursive= recursive
        self._softmaxLayer= torch.nn.Softmax(dim=1)
        self._sampleAsState= sampleAsState
    
    def reset_probability(self, population):
        #self._state= torch.stack((population["S"], population["E"], population["I"], population["R"]))
        self._P= None


    def forward(self, state):

        if self._train==False:
            psMatrix= self._psMatrix
        else:
            psMatrix= self.get_psMatrix()
        
        logProb= 1-self._IS*state[]

        L= torch.zeros_like(state, device=device)

        L[0]= state[0]*torch.matmul(self._IS, state[2])
        L[1]= torch.matmul(state[1], self._EE)
        L[2]= torch.matmul(state[2], self._II)
        L[3]= torch.matmul(state[3], self._RR)
        prob= 1- torch.exp(torch.matmul(torch.log(1 - psMatrix.T), L))
    
        _stable_prob= 1- torch.sum(prob, 0)
        prob= prob+ state*_stable_prob
        self._P= prob

        if self._sampleAsState:
            stateReturn= self.sample_uniform_matrix(prob)
        else:
            stateReturn= prob     

        return stateReturn, self._P
        #return torch.sum(self._state, 1), torch.sum(prob, 1), prob
    
    def get_population_num(self, state):

        return torch.sum(state, 1)

    #sample nxm pobability matrix, of 0 dimension, which contains n choise for a random variable
    def sample_uniform_matrix(self, P):  
        state= torch.zeros_like(P, device=device)
        U= torch.rand(self._n).to(device)
        for i in range(P.size()[0]):
            U= U- P[i]
            state[i]= U<0
            U= U+state[i]
        return state
    
    def get_population_dir(self, state):
        return { #a columnar DB somewhat reminds the Pandas DataFrame 
                "S": state[0],
                "E": state[1],
                "I": state[2],
                "R": state[3],
                }
    
    def get_probability(self):
        return self._P
    
    def get_probability_transition_matrix(self):
        
        return self._psMatrix
    
    def get_psMatrix(self):
        #psMatrix= torch.sigmoid(self._psMatrix)
        psMatrix= self._softmaxLayer(self._psMatrix)
        psMatrix= psMatrix*self._forceCc
        if self._cc is not None:
                psMatrix= psMatrix*self._cc
        return psMatrix
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
    def __init__(self, ISNet, psMatrix, device, train=False, cc=None, sampleAsState=True):
        super(pure_EPI_dense, self).__init__()
        self._n= ISNet.size()[0]
        self._IS= ISNet[None, :]
        self._psMatrix= psMatrix
        self._train= train
        if train==True:
            self._psMatrix=torch.nn.Parameter(psMatrix)
        self._forceCc=torch.tensor([[0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0]], device=device) 
        self._cc= cc
        self._softmaxLayer= torch.nn.Softmax(dim=1)
        self._sampleAsState= sampleAsState


    def forward(self, state):
        psMatrix= self.get_psMatrix()
        ps10= torch.zeros_like(state, device=device)#1*4*1000
        logProbIS= torch.zeros([state.shape[0], self._n, self._n], device=device)
        for i in range(state.shape[0]):
            logProbIS[i, :]= torch.log(1-self._IS*state[None, i, 0].T*state[i, 2]*psMatrix[0, 1])#2*1000*1000, 2*1000, 1000*2!
    
        ps10[:, 1]= 1- torch.exp(torch.sum(logProbIS, dim=2))
        ps10[:, 0]= 1- ps10[:, 1]
        constantM=torch.tensor([[0, 0, 0, 0],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1],
                                [1, 1, 1, 1]], device=device) 
        ps2T= (psMatrix*constantM).T
        for i in range(state.shape[0]):
            state[i, :]= state[i, 0, :]*ps10[i, :]+ ps2T.matmul(state[i, :])

        if self._sampleAsState:
            return self.sample_uniform_matrix(state)

        return state
    
    def get_population_num(self, state):

        return torch.sum(state, 2)

    #sample nxm pobability matrix, of 0 dimension, which contains n choise for a random variable
    def sample_uniform_matrix(self, P):  
        state= torch.zeros_like(P, device=device)
        U= torch.rand(self._n).to(device)
        for i in range(P.size()[1]):
            U= U- P[:, i]
            state[:, i]= U<0
            U= U+state[:, i]
        return state
    
    def get_population_dir(self, state):
        return { #a columnar DB somewhat reminds the Pandas DataFrame 
                "S": state[0],
                "E": state[1],
                "I": state[2],
                "R": state[3],
                }
    
    def get_original_transition_matrix(self):
        
        return self._psMatrix
    
    def expand_psMatrix(self, psMatrix):
        stabelProb= 1- torch.sum(psMatrix, 1)
        psSize= psMatrix.shape[0]
        eyeM= torch.eye(psSize, device=device)
        return eyeM*stabelProb+psMatrix

    
    def get_psMatrix(self):
        if self.train== True:
            psMatrix= self._softmaxLayer(self._psMatrix)
            psMatrix= psMatrix*self._forceCc
            if self._cc is not None:
                    psMatrix= psMatrix*self._cc
            return self.expand_psMatrix(psMatrix)
        else:
            return self.expand_psMatrix(self._psMatrix)
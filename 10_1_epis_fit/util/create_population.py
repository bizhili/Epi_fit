import torch
import random


def population(n:int, device= "cpu"):
  firstInfected= random.randint(0, n-1)
  initial_population = { #a columnar DB somewhat reminds the Pandas DataFrame 
      "S":torch.ones(n).to(device),
      "E":torch.zeros(n).to(device),
      "I":torch.zeros(n).to(device),
      "R":torch.zeros(n).to(device),
  }
  initial_population["S"][firstInfected]=0 
  initial_population["I"][firstInfected]=1 #first one infected
  state= torch.stack((initial_population["S"], initial_population["E"], initial_population["I"], initial_population["R"]))
  return state[None, ...]

def populations(n: int, number: int, device= "cpu"):
    populations=[]
    for i in range(number):
       populations.append(population(n, device=device))
    return torch.stack(populations)

def static_population(n:int, firstT:int, device= "cpu"):
  initial_population = { #a columnar DB somewhat reminds the Pandas DataFrame 
      "S":torch.zeros(n).to(device),
      "E":torch.zeros(n).to(device),
      "I":torch.zeros(n).to(device),
      "R":torch.zeros(n).to(device),
  }
  initial_population["R"][0]=-firstT-1+1e-10#first one infected
  state= torch.stack((initial_population["S"], initial_population["E"], initial_population["I"], initial_population["R"]))
  return state[None, ...]

def static_populations(n: int, firstTS: list, device= "cpu"):
    populations=[]
    for _, j in enumerate(firstTS):
       populations.append(static_population(n, j, device=device))
    return torch.stack(populations)

def general_populations(kn:int, n:int, device="cpu"):
    state= torch.zeros([1, kn, n, 4, 1], device=device)# 3,1000,4,1
    state[0, :, :, 0, 0]= 1
    return state
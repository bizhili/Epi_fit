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
  return initial_population

def populations(n: int, number: int, device= "cpu"):
    populations=[]
    for i in range(number):
       populations.append(population(n, device=device))
    return populations
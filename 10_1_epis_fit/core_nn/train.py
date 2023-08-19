import sys
import torch
if __name__== "__main__":
    sys.path.append('../')
import core_nn.general_EPIs_dense as general_EPIs_dense
import util.population as population
import core_nn.sim as sim

def generate_dateset(model: general_EPIs_dense.general_EPIs_dense, ys, timeHorizon=100, windowSize= 5, Kn= 2, category= 0, device="cpu"):
    n= model._n
    with torch.no_grad():
        statesZero= population.general_populations(Kn, n, device=device)# 3, 1000, 4, 1
        population.init_first_cases(statesZero)
        allStates, newI= sim.daily_new_cases(timeHorizon, model, statesZero, category=category, device=device)#generate data more times, 600, 3, 1000, 4, 1
        diffI= torch.abs(newI- ys)
        middleValue= torch.median(diffI)
        meanValue= torch.mean(diffI)
        biggerThanMean= (diffI-meanValue)>0
        biggerThanMeanNum= biggerThanMean.sum()
        thresholdI= 0 #if biggerThanMeanNum< leastSampleNum else meanValue
        infectiveNum= []
        usefulStates= []
        for i in range(timeHorizon-windowSize):
            timesTemp= torch.tensor([j for j in range(i, i+windowSize)], dtype=torch.float32, device=device)
            embedTemp= torch.stack([ys[i+1: i+windowSize+1], timesTemp])# +1 means predict
            if diffI[i]>= thresholdI:
                infectiveNum.append(embedTemp)
                usefulStates.append(allStates[i, ...])
        infectiveNum= torch.stack(infectiveNum)
        usefulStates= torch.stack(usefulStates)
        usefulStates= usefulStates.squeeze(dim=1)
        dataset = torch.utils.data.TensorDataset(usefulStates, infectiveNum)
        return dataset, newI
    


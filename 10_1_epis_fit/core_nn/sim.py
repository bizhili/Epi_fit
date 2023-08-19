import sys
import torch
if __name__== "__main__":
    sys.path.append('../')
import core_nn.general_EPIs_dense as general_EPIs_dense


def daily_new_cases(timeHorizon, model: general_EPIs_dense.general_EPIs_dense, state, category=0,  device="cpu"):
    with torch.no_grad():
        stateHistory=[]
        staticNewI= []
        for i in range(timeHorizon):
            t= torch.tensor(i, dtype=torch.float32, device=device)
            t= torch.stack([t])
            t= t[..., None]
            state, newI= model(state, t, category= category)
            stateHistory.append(state)
            KnT= newI.shape[1]
            oneT= 1
            for j in range(KnT):
                oneT*= 1- newI[0, j, :]
            newI= 1- oneT
            staticNewI.append(newI.sum())
        stateHistory= torch.stack(stateHistory)
        staticNewI= torch.stack(staticNewI)
        return stateHistory, staticNewI
    
def active_cases(timeHorizon, model: general_EPIs_dense.general_EPIs_dense, state, category=0, device="cpu"):
    with torch.no_grad():
        stateHistory=[]
        staticSI=[]
        for i in range(timeHorizon):
            t= torch.tensor(i, dtype=torch.float32, device=device)
            t= torch.stack([t])
            t= t[..., None]
            state, _ = model(state, t, category= category)
            stateHistory.append(state)
            KnT= state.shape[1]
            oneT= 1
            for j in range(KnT):
                oneT*= 1- state[0, j, :, 2, 0]
            statisticI= 1- oneT
            staticSI.append(statisticI.sum())
        stateHistory= torch.stack(stateHistory)
        staticSI= torch.stack(staticSI)
        return stateHistory, staticSI

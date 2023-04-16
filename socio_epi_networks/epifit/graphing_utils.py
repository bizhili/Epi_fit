import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def plot(log, colors, age_sizes, group_names=None, state_compartments=None, granularity="individual", by_age=True, title="", percent_extremes_remove=0.1): 
    """
    Generalized plotting function - can plot by age groups and states.
    
    parameters: Tensor, list, list, list, dictionary, string, float
    
    """
    if state_compartments is None:
        state_compartments = {"all states": range(len(colors))}

    repeats = log.size(dim=0)

    def plot_between_data_points(start, stop, i=0, title=title):
        age_group_idx = torch.LongTensor(range(start,stop)).cpu()
        if granularity == 'individual':
            stats = log.cpu().index_select(dim=3,index=age_group_idx).sum(dim=3)
        elif granularity == 'age_group':
            stats = log.cpu()[:, :, :, i]
        mean = torch.mean(stats,dim=0).t()
        std = torch.std(stats,dim=0).t()
        low = torch.kthvalue(stats,int(percent_extremes_remove*repeats)+1,dim=0)[0].t()
        high = torch.kthvalue(stats,int((1-percent_extremes_remove)*repeats)+1,dim=0)[0].t()

        fig, axes = plt.subplots(len(state_compartments), figsize=(10, 3 * len(state_compartments)), sharex=True)

        if len(state_compartments) == 1:
            axes = [axes]

        for (comp_name, compartment), ax in zip(state_compartments.items(), axes):
            for i in compartment:
                c,l = colors[i]
                ax.fill_between(np.arange(0,mean.size()[1]),low[i],high[i],alpha=0.1,color=c)
                ax.plot(mean[i],label=l,color=c)
                ax.set_title(f'{comp_name} {title}')
                ax.legend()

        plt.show()

    plot_between_data_points(0,log.size(dim=3))
                             
    if not by_age:
        return
    
    # results by age group
    age_idx = [0]+age_sizes.cumsum(dim=0).tolist()
    age_idx = [(x,y) for x,y in zip(age_idx[:-1],age_idx[1:])]

    for i in range(len(age_idx)):
        x,y = age_idx[i]
        group_name = group_names[i]
        plot_between_data_points(x, y, i, f'- age group {group_name}')
            

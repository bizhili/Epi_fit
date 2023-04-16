import torch
import torch.nn.functional as F
import itertools
import random
import math
device = "cuda" if torch.cuda.is_available() else "cpu"

class S_I(torch.nn.Module):
    def __init__(self, contact_net, pinf, I_name, susceptiveness_name, infectiveness_name, start=0):
        super(S_I,self).__init__()
        self._contact_net, self._pinf, self._n, self._t, self._start = contact_net, pinf, contact_net.size()[0], 0, start
        self._states_names = {'I': I_name, 'susceptiveness': susceptiveness_name, 'infectiveness': infectiveness_name}
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            I = states[self._states_names['I']] # boolean state of being infected
            susceptiveness = states[self._states_names['susceptiveness']] # floating point, number between 0 and 1 means probability of being infected
            infectiveness = states[self._states_names['infectiveness']] # floating point, number between 0 and 1 means probability of infecting another
            
            susceptible = (I==0).float()
            susceptible*=susceptiveness
            infective=infectiveness*I

            srcidx, dstidx = self._contact_net.coalesce().indices()
            values = susceptible.index_select(dim=0, index=srcidx)*infective.index_select(dim=0, index=dstidx)
            values = torch.log(1 - values) # log of probability of not getting infected
            con = torch.sparse.FloatTensor(indices=torch.stack((srcidx, dstidx)), values=values, size=(self._n, self._n)) # note that the log of 1 is 0 so sparse matrix is appropriate
            
            dI = 1 - torch.exp(con.mm(torch.ones(self._n).to(device).unsqueeze(dim=1)).squeeze()) # probability of being infected that day
            dI = torch.rand(self._n).to(device) < dI
            I = torch.max(I, dI) # got infected
            states[self._states_names['I']] = I
        return states
    
class S_E(torch.nn.Module):
    def __init__(self, contact_net, pinf, E_name, susceptiveness_name, infectiveness_name, incubation_name, infective_name=None, start=0):
        super(S_E, self).__init__()
        self._contact_net, self._pinf, self._n, self._t, self._start = contact_net, pinf, contact_net.size()[0], 0, start
        self._states_names = {'E': E_name, 'susceptiveness': susceptiveness_name, 'infectiveness': infectiveness_name, 'incubation': incubation_name}
        if infective_name: self._states_names['infective'] = infective_name # if those that are infective are more selective than just having E == 0
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            E = states[self._states_names['E']] # int state of how much time left for incubation (until fully infected). inf->not infected at all, 0->infective, other->incubation
            susceptiveness = states[self._states_names['susceptiveness']] # floating point, number between 0 and 1 means probability of being infected
            infectiveness = states[self._states_names['infectiveness']] # floating point, number between 0 and 1 means probability of infecting another
            incubation = states[self._states_names['incubation']] # floating point, number at least 0 means time for person to become fully infected after exposure

            E = F.relu(E-1)
            infective = (E==0).float() if 'infective' not in self._states_names else states[self._states_names['infective']]
            susceptible = (E==math.inf).float()
            susceptible*=susceptiveness
            infective*=infectiveness

            srcidx, dstidx = self._contact_net.coalesce().indices()
            values = susceptible.index_select(dim=0, index=srcidx)*infective.index_select(dim=0, index=dstidx)
            values = torch.log(1 - values) # log of probability of not getting infected
            con = torch.sparse.FloatTensor(indices=torch.stack((srcidx, dstidx)), values=values, size=(self._n, self._n)) # note that the log of 1 is 0 so sparse matrix is appropriate
            
            dI = 1 - torch.exp(con.mm(torch.ones(self._n).to(device).unsqueeze(dim=1)).squeeze()) # probability of being infected that day
            dI = torch.rand(self._n).to(device) < dI
            E = torch.where(dI, incubation, E) # got infected
            states[self._states_names['E']] = E
        return states
    
class E_I(torch.nn.Module):
    def __init__(self, I_name, E_name, start=0):
        super(E_I, self).__init__()
        self._states_names = {'E': E_name, 'I': I_name}
        self._t, self._start = 0, start
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            states[self._states_names['I']] = (states[self._states_names['E']]==0).float()
        return states
    
class I_S(torch.nn.Module):
    def __init__(self, contact_net, I_name, R_name, recover_time_name, start=0):
        super(I_S,self).__init__()
        self._contact_net, self._n, self._t, self._start = contact_net, contact_net.size()[0], 0, start
        self._states_names = {'I': I_name, 'R': R_name, 'recover_time': recover_time_name}
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            R = states[self._states_names['R']]
            I = states[self._states_names['I']]
            recover_time = states[self._states_names['recover_time']]
            R = F.relu(R-1)
            R = torch.where((I==1) & (R==math.inf), recover_time, R) # setting recovery time for those newly infected  
            I = torch.where((R==0), torch.zeros(self._n).to(device), I)
            R = torch.where((R==0), torch.ones(self._n).to(device)*math.inf, R) # reset recovery times for those who are susceptible again
            states[self._states_names['R']] = R
            states[self._states_names['I']] = I
        return states
    
class I_R(torch.nn.Module):
    def __init__(self, I_name, R_name, recover_time_name, start=0):
        super(I_R,self).__init__()
        self._states_names = {'I': I_name, 'R': R_name, 'recover_time': recover_time_name}
        self._t, self._start = 0, start
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            R = states[self._states_names['R']]
            I = states[self._states_names['I']]
            recover_time = states[self._states_names['recover_time']]
            n = R.size()[0]

            R = F.relu(R-1)
            R = torch.where((I==1) & (R==math.inf), recover_time, R) # setting recovery time for those newly infected  
            I = torch.where((R==0), torch.zeros(n).to(device), I)

            states[self._states_names['R']] = R
            states[self._states_names['I']] = I
        return states
    

class I_D(torch.nn.Module):
    def __init__(self, I_name, D_name, fatality_name, start=0):
        super(I_D, self).__init__()
        self._states_names = {'I': I_name, 'D': D_name, 'fatality': fatality_name}
        self._t, self._start = 0, start
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            D = states[self._states_names['D']]
            I = states[self._states_names['I']]
            fatality = states[self._states_names['fatality']]
            n = D.size()[0]

            D = F.relu(D-1)
            D = torch.where((I==1) & (D==math.inf), fatality, D) # setting fatality for those newly infected  
            I = torch.where((D==0), torch.zeros(n).to(device), I)

            states[self._states_names['D']] = D
            states[self._states_names['I']] = I
        return states
    
class ControllingState_ControlledState(torch.nn.Module):
    def __init__(self, controlling_state_name, controlled_state_name, controlling_state_value, controlled_state_value, pinf, required_initial_controlledstatevalue=None, maxcontrolled=math.inf, controlledval_is_multiplier=False, start=0):
        super(ControllingState_ControlledState, self).__init__()
        self._controlling_state_value = controlling_state_value
        self._controlled_state_value = controlled_state_value # the value that people with controlling_state_value in controlling_state will take on in controlled_state
        self._pinf = pinf # the probability of a person's controlled_state to follow what they are in controlling_state
        self._maxcontrolled = maxcontrolled # the maximum people in controlled state that can be switched in a given day by controlling_state
        self._multiplier = controlledval_is_multiplier
        self._required_initial_controlledstatevalue = required_initial_controlledstatevalue
        self._states_names = {"controlling_state": controlling_state_name, "controlled_state": controlled_state_name}
        self._t, self._start = 0, start
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            controlling_state = states[self._states_names['controlling_state']]
            controlled_state = states[self._states_names['controlled_state']]
            n = controlled_state.size()[0]
            indices_to_change = ((torch.rand(n).to(device)<(torch.ones(n).to(device)*self._pinf))&(controlling_state==self._controlling_state_value))
            indices_to_change = indices_to_change&(controlled_state!=self._controlled_state_value)
            if self._required_initial_controlledstatevalue is not None: indices_to_change = indices_to_change&(controlled_state==self._required_initial_controlledstatevalue)
            if indices_to_change.sum().item() > self._maxcontrolled: 
                would_change = indices_to_change.nonzero().flatten().tolist()
                indices_to_change = random.sample(would_change, self._maxcontrolled)
            if self._multiplier: controlled_state[indices_to_change]*=self._controlled_state_value
            else: controlled_state[indices_to_change] = self._controlled_state_value
            states[self._states_names['controlled_state']] = controlled_state
        return states

#initialize state with name and state value dictionary, time is 0, and set simulation start time
class InitializeValues(torch.nn.Module):
    def __init__(self, state_name, state_value, start=0):
        super(InitializeValues, self).__init__()
        self._state_name, self._state_value, self._t, self._start = state_name, state_value, 0, start
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            states[self._state_name][:] = self._state_value
        return states

#increase or decrease _state_name by _inc_dec_value
class Inc_Dec(torch.nn.Module):
    def __init__(self, state_name, inc_dec_value=1, inc=True, start=0):
        super(Inc_Dec, self).__init__()
        self._state_name, self._inc_dec_value, self._inc, self._t, self._start = state_name, inc_dec_value, inc, 0, start
    
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            if self._inc: states[self._state_name]+=self._inc_dec_value
            else: states[self._state_name]-=self._inc_dec_value
        return states

#the change of contact effects the change of susceptible_state (susceptive probability)
class Contacts_InfectingState(torch.nn.Module):
    def __init__(self, contact_net, contacts_state_name, susceptible_state_name, pinf, infected_value, start=0):
        super(Contacts_InfectingState, self).__init__()
        self._contact_net, self._pinf, self._n, self._t, self._start = contact_net, pinf, contact_net.size()[0], 0, start
        self._infected_value = infected_value # the value people "infected" by this transition take on in susceptible_state
        
        # contacts can "infect" and change the state of someone in susceptible_state to be infected_value
        self._states_names = {"contacts_state": contacts_state_name, "susceptible_state": susceptible_state_name}
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            contacts_state = states[self._states_names['contacts_state']]
            susceptible_state = states[self._states_names['susceptible_state']]

            dI = self._contact_net.mm(contacts_state.unsqueeze(dim=1))
            dI = 1 - (1-self._pinf)**dI.squeeze()
            dI = torch.rand(self._n).to(device) < dI

            susceptible_state = torch.where(dI, torch.ones(self._n).to(device)*self._infected_value, susceptible_state)

            states[self._states_names['susceptible_state']] = susceptible_state
        return states

#? 
class Contacts_SwitchState(torch.nn.Module):
    def __init__(self, preevaluate, contact_net, state_name, active_state_name=None, start=0, requirement_forinfluencers=None):
        super(Contacts_SwitchState, self).__init__()
        self._contact_net, self._preevaluate, self._n, self._t, self._start = contact_net, preevaluate, contact_net.size()[0], 0, start
        self._states_names = {"state": state_name} # the state in which contacts can influence their contacts to switch to or stay in their state
        if active_state_name: self._states_names["active_state"] = active_state_name # the state which tells which people can be influenced and can influence for state - the ones without a zero in active state
        self._req_infl = requirement_forinfluencers # if this is not None, then it is a function which can be applied on the state to judge who can influence
        
    def forward(self, states):
        self._t+=1
        if self._t >= self._start:
            state = states[self._states_names['state']]
            active_state = states[self._states_names['active_state']] if 'active_state' in self._states_names else torch.ones(self._n).to(device)

            influencers = self._req_infl(states) if self._req_infl is not None else torch.ones(self._n).bool().to(device)
            active = (active_state!=0) # those who are not active cannot influence or be influenced
            idx_reevaluate = torch.rand(self._n).to(device)<self._preevaluate # the people who could change their mind
            idx_reevaluate = (idx_reevaluate&active)
            contacts_instate = self._contact_net.mm(((state==1)&active&influencers).float().unsqueeze(dim=1)).squeeze() # number of contacts in the state (value 1)
            total_contacts = self._contact_net.mm((active&influencers).float().unsqueeze(dim=1)).squeeze()
            choose_state = torch.rand(self._n).to(device) < (contacts_instate/total_contacts) # if a person has A contacts in the state and B contacts not, then there is probability A/(A+B) that this person will choose state A (symmetric is true with B)
            state[(idx_reevaluate&choose_state)] = 1 
            state[(idx_reevaluate&(~choose_state))] = 0 
        return states

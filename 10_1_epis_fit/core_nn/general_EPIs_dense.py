
import torch

class general_EPIs_dense(torch.nn.Module):
    def __init__(self, contact, psMatrices, crossImmMatrix, Ts, train=False, sampleAsState=False, device="cpu"):
        super(general_EPIs_dense, self).__init__()
        self._n= contact.size()[1]
        if len(psMatrices.size())==3:
            psMatrices= psMatrices[None, ...]
        self._k= psMatrices.size()[1]#m, 1, 4, 4
        self._contact= contact[None, None, ...]
        self._psMatrices= psMatrices
        self._crossImmMatrix= crossImmMatrix
        self._train= train
        self._device= device
        self._sampleAsState= sampleAsState
        self._Ts= Ts[None, ...]
        self._mySig = torch.nn.Sigmoid()
        self._mySig2 = torch.nn.Sigmoid()
        self._myrelu= torch.nn.ReLU()
        self._mysoftmax= torch.nn.Softmax(dim= 2)
        self._mysoftPlus= torch.nn.Softplus()
        self._trust = torch.tensor([[10]*self._k], device=self._device) 
        self._sampleT= None
        self.beta= 1
        if train==True:
            self._psMatrices= torch.nn.Parameter(self._psMatrices)
            self._crossImmMatrix= torch.nn.Parameter(self._crossImmMatrix)
            self._Ts= torch.nn.Parameter(self._Ts)
            self._trust = torch.tensor([[0.8]*self._k], device=self._device) 
            self._trust= torch.nn.Parameter(self._trust)
            self._contact= torch.nn.Parameter(self._contact)
        self._init_mask_T()
    def forward(self, state, t, category=0): # state shape:(1, K, n, 4, 1), t:(1, 1)
        self._sampleT= self._mySig(self._trust*(t-self._Ts[:, category, ...]))
        cL= self._sampleT[..., None, None]
        cL= cL*self._controlMask+self._controlI
        cL= cL[:, None, :, None, ..., None]
        Linear= self.encode_linear(category)
        linear= Linear*cL
        A= linear.matmul(state[:, :, None, :, None, ...])
        Psts= torch.prod(A, dim=1)
        Psts= Psts.squeeze(dim=-1)
        sumPsts= Psts.sum(dim= -2)+ 1e-8
        sumPsts= sumPsts[:, :, :, None, :]
        Psts= Psts/sumPsts
        Psts2= Psts[:, :, :, 1, 0]
        s= state[:, :, :, 0:1, 0]
        i= state[:, :, :, 2:3, 0]
        s= s.permute(0, 1, 3, 2)
        Temp= self._myrelu(self._contact+self._contact.transpose(2,3))/2*i*s
        Psts2= 1-Temp*Psts2 [..., None]
        Psts2= (1-torch.prod(Psts2, dim=-2))/(s.squeeze(2)+1e-8)
        #
        PstsNew= Psts.clone()
        #
        PstsNew[:, :, :, 1, 0]= Psts2
        PstsNew[:, :, :, 0, 0]= 1- Psts2

        PstsE2I= PstsNew[..., 2, 1]#1, 3, 1000, 1
        e= state[..., 1, 0]#1, 3, 1000, 1
        newI= PstsE2I*e
        
        newState= PstsNew.matmul(state)# new state shape:(K, n, 4, 1)
        return newState, newI
        
    def _init_mask_T(self):
        self._mask1= torch.tensor([ [0, 0, -100, -100],
                                    [-100, 0, 0, -100],
                                    [-100, -100, 0, 0],
                                    [-100, -100, -100, 0]], device=self._device) 
        self._mask2= torch.tensor([ [0, 1, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]], device=self._device) 
        self._T    = torch.tensor([ [0, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]], device=self._device) 
        self._mask1= self._mask1[None, ...]
        self._controlMask= torch.tensor([   [0, 1, 1, 1],
                                            [1, 0, 1, 1],
                                            [1, 1, 0, 1],
                                            [1, 1, 1, 0]], device=self._device)
        self._controlI= torch.stack([1-self._controlMask]*self._k)
        self._controlI= self._controlI[None, ...]
        
    def encode_linear(self, category= 0):
        ts= 4
        linear= torch.ones(self._k, self._k, 1, ts, ts, ts, device=self._device)
        psMatrix= self._mysoftmax(self._psMatrices[category, ...]+self._mask1)
        crossImmMatrix= (self._crossImmMatrix+self._crossImmMatrix.T)/2
        for i in range(self._k): # k layer
            for j in range(self._k):# prod layer
                if i==j: #load paMatrices
                    for k in range(ts):
                        linear[j, i, 0, :, :, k]= psMatrix[i, ...].T
                else: #load crossImmMatrix
                    linear[j, i, 0, 1, 0, 2:4]= crossImmMatrix[j, i]
        return linear[None, ...]

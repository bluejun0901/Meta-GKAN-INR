import torch
from torch import nn
from skimage.metrics import peak_signal_noise_ratio
import math
dtype = torch.cuda.FloatTensor

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=3):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)    
        
    def forward(self, input):
        return self.linear(torch.sin(self.omega_0 * input))
    
class GKANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features*3, out_features)  
        self.omega_0 = torch.Tensor([[1],[2],[3]]).type(
            dtype).repeat(1, in_features).reshape(1, in_features*3)
    def forward(self, input):
        return self.linear(torch.sin(self.omega_0*input.repeat(1,3)))

class input_mapping(nn.Module): 
    def __init__(self, B):
        super(input_mapping, self).__init__()
        self.B = B
    def forward(self, input):
        x_proj = (input) @ self.B.t()
        in_data = torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)
        return in_data
    
class INR(nn.Module): 
    def __init__(self,method):
        super(INR, self).__init__()
        
        if method == 'SIREN':
            mid = 450
            self.H_net = nn.Sequential(nn.Linear(3,mid),
                                        SineLayer(mid,mid),
                                        SineLayer(mid,mid),
                                        SineLayer(mid,1))
        elif method == 'PE':
            mid = 320
            self.B_gauss = torch.Tensor(mid,3).type(dtype)
            torch.nn.init.kaiming_normal_(self.B_gauss, a=math.sqrt(2))
            self.H_net = nn.Sequential(input_mapping(self.B_gauss),
                                    nn.Linear(2*mid,mid),
                                    nn.ReLU(),
                                    nn.Linear(mid,mid),
                                    nn.ReLU(),
                                    nn.Linear(mid,mid),
                                    nn.ReLU(),
                                    nn.Linear(mid,1))
            
        elif method == 'GKAN':
            mid = 260
            self.H_net = nn.Sequential(nn.Linear(3,mid),
                                        GKANLayer(mid,mid),
                                        GKANLayer(mid,mid),
                                        GKANLayer(mid,1))
        
    def forward(self, input):
        return (self.H_net(input))
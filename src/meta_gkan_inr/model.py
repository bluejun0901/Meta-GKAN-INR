import torch
from torch import nn
import math

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
        self.linear = nn.Linear(in_features * 3, out_features)
        base = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        omega = base.repeat(1, in_features).reshape(1, in_features * 3)
        self.register_buffer("omega_0", omega)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        omega_key = prefix + "omega_0"
        if omega_key not in state_dict:
            # Legacy checkpoints may omit omega_0; keep the existing buffer.
            state_dict[omega_key] = self.omega_0.detach().clone().cpu()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    def forward(self, input):
        return self.linear(torch.sin(self.omega_0 * input.repeat(1, 3)))

class input_mapping(nn.Module): 
    def __init__(self, B):
        super(input_mapping, self).__init__()
        self.register_buffer("B", B)
    def forward(self, input):
        x_proj = input @ self.B.t() # type: ignore
        in_data = torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)
        return in_data
    
class INR(nn.Module): 
    def __init__(self,method):
        super(INR, self).__init__()
        self.method = method
        
        if method == 'SIREN':
            mid = 450
            self.H_net = nn.Sequential(nn.Linear(3,mid),
                                        SineLayer(mid,mid),
                                        SineLayer(mid,mid),
                                        SineLayer(mid,1))
        elif method == 'PE':
            mid = 320
            B_gauss = torch.empty(mid, 3, dtype=torch.float32)
            torch.nn.init.kaiming_normal_(B_gauss, a=math.sqrt(2))
            self.H_net = nn.Sequential(input_mapping(B_gauss),
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

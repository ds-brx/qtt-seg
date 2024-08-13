
import torch
import torch.nn as nn

from models import MLP, ConvNet


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        config_in=5,
        config_out=32,
        curve_in=50,
        curve_out=16,
        meta_in=4,
        meta_out=16,
        mlp_out=32
    ):
        super().__init__()

        enc_dims = 0 

        # config encoder
        self.config_encoder = MLP(config_in, config_out, 3, 128)
        enc_dims += config_out 

        # budget encoder
        enc_dims += 1

        # curve encoder
        self.curve_embedder = ConvNet(curve_in, curve_out)
        enc_dims += curve_out

        # meta encoder
        self.fc_meta = nn.Linear(meta_in, meta_out)
        enc_dims += meta_out

        self.head = MLP(enc_dims, mlp_out, 3, 128, act_fn=nn.GELU)

    def forward(self, config, curve, budget, metafeat):
        x = []
        
        config_enc = self.config_encoder(config)
        x = torch.cat([torch.tensor(x),config_enc], dim=1)
        x = torch.cat([x, budget], dim=1)
       
        curve = curve.unsqueeze(0).permute(1,2,0)
        curve_enc = self.curve_embedder(curve.float())
        x = torch.cat([x, curve_enc], dim=1)
        
        meta_enc = self.fc_meta(metafeat.float())
        x = torch.cat([x, meta_enc], dim=1)
        x = self.head(x)
        return x


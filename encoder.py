
import torch
import torch.nn as nn

from models import MLP, ConvNet


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        config_in=5,
        config_out=8,
        curve_in=10,
        curve_out=8,
        meta_in=4,
        meta_out=8,
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

        self.head = MLP(enc_dims, mlp_out, 3, 16, act_fn=nn.GELU)

    def forward(self, config, curve, budget, metafeat):
        # Collect the features
        x = []

        # Encode configuration
        config_enc = self.config_encoder(config)
        x.append(config_enc)
        
        # Add budget
        x.append(budget)

        # Encode curve
        curve = curve.unsqueeze(0).permute(1, 2, 0)
        curve_enc = self.curve_embedder(curve.float())
        x.append(curve_enc)
        
        # Encode meta-features
        meta_enc = self.fc_meta(metafeat.float())
        x.append(meta_enc)
        
        # Concatenate all encoded features along dimension 1
        x = torch.cat(x, dim=1)
        
        # Pass through the final MLP head
        x = self.head(x)

        return x

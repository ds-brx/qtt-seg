import torch
import torch.nn as nn
from encoder import FeatureEncoder

class CostEstimator(torch.nn.Module):
    refine_steps = 50
    refine_lr = 1e-3

    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.encoder = FeatureEncoder(mlp_out=1).to(self.device)  # Move the encoder to the specified device
        self.prelu = nn.PReLU().to(self.device)  # Define PReLU layer and move it to the specified device

    def fit_pipeline(self, config, curve, budget, target, meta_feat):
        optimizer = torch.optim.Adam(self.parameters(), self.refine_lr)
        criterion = torch.nn.MSELoss()
        self.train()

        # Ensure the target is moved to the correct device
        target = target.to(self.device)

        for i in range(self.refine_steps):
            optimizer.zero_grad()

            # Move input tensors to the correct device
            output = self.encoder(config, curve, budget, meta_feat).to(self.device)

            loss = criterion(output, target.float())
            loss.backward()
            optimizer.step()
            print("Step: {} Loss: {}".format(i, loss.item()))

    def predict_pipeline(self, config, curve, budget, meta_feat):
        self.eval()

        with torch.no_grad():
            # Move input tensors to the correct device
            output = self.encoder(config, curve, budget, meta_feat).to(self.device)

            # Apply PReLU activation

        return output

from encoder import FeatureEncoder

import torch


class CostEstimator(torch.nn.Module):
    refine_steps = 50
    refine_lr = 1e-3

    def __init__(self):
        super().__init__()
        self.encoder = FeatureEncoder(mlp_out=1)

    def fit_pipeline(self, config,curve,budget,target,meta_feat):
        optimizer = torch.optim.Adam(self.parameters(), self.refine_lr)
        criterion = torch.nn.MSELoss()
        self.train()
        for i in range(self.refine_steps):
            optimizer.zero_grad()
            feat = self.encoder(config,curve,budget,meta_feat)
            output = torch.nn.functional.relu(feat)
            loss = criterion(output,target.float())
            loss.backward()
            optimizer.step()
            print("Step: {} Loss: {}".format(i,loss))

    def predict_pipeline(
        self,
        config,curve,budget,meta_feat
    ):
        self.eval()

        with torch.no_grad(): 
            feat = self.encoder(config,curve,budget,meta_feat)
            output = torch.nn.functional.relu(feat)

        return output

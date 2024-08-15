import gpytorch
import torch
from encoder import FeatureEncoder

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar) 


class PerfEstimator(torch.nn.Module):
    refine_steps = 50
    refine_lr = 1e-3

    def __init__(
        self,
        device=torch.device("cpu")  # Default to CPU; pass in 'cuda' for GPU
    ):
        super().__init__()

        self.device = device

        # Move the encoder and GP model to the specified device
        self.encoder = FeatureEncoder(mlp_out=32).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gp_model = GPRegressionModel(
            train_x=None,
            train_y=None,
            likelihood=self.likelihood,
        ).to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood,
            self.gp_model,
        ).to(self.device)

    def fit_pipeline(self, config, curve, budget, target, meta_feat):
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), self.refine_lr)

        # Ensure that all tensors are moved to the appropriate device
        target = target.to(self.device)
        print(target)
        print(curve)

        for i in range(self.refine_steps):
            optimizer.zero_grad()
            feat = self.encoder(config, curve, budget, meta_feat).to(self.device)
            self.gp_model.set_train_data(feat, target, False)
            output = self.gp_model(feat)
            print(output)
            loss = -self.mll(output, target)
            loss.backward()
            optimizer.step()
            print("Step: {} Loss: {}".format(i, loss.item()))

        self.eval()

    def predict_pipeline(
        self,
        config, curve, budget, meta_feat,
    ):  
        print(curve)
        self.eval()
        with torch.no_grad(): 
            test_x = self.encoder(config, curve, budget, meta_feat).to(self.device).float()
            pred = self.likelihood(self.gp_model(test_x))
        mean = pred.mean.reshape(-1).to(self.device)
        std = pred.stddev.reshape(-1).to(self.device)
        print(mean,std)
        return mean, std

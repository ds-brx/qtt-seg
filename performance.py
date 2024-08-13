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
        self
    ):
        super().__init__()

        self.encoder = FeatureEncoder(mlp_out=32)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = GPRegressionModel(
            train_x=None,
            train_y=None,
            likelihood=self.likelihood,
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood,
            self.gp_model,
        )
    def fit_pipeline(self, config,curve,budget,target,meta_feat):
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), self.refine_lr)


        for i in range(self.refine_steps):
            optimizer.zero_grad()
            feat = self.encoder(config,curve,budget,meta_feat)
            self.gp_model.set_train_data(feat, target, False)
            output = self.gp_model(feat)
            loss = -self.mll(output, target) 
            loss.backward()
            optimizer.step()
            print("Step: {} Loss: {}".format(i,loss))

        self.eval()
    def predict_pipeline(
        self,
        config,curve,budget,meta_feat,
    ):
        self.eval()
        with torch.no_grad(): 
            test_x = self.encoder(config,curve,budget,meta_feat)
            pred = self.likelihood(self.gp_model(test_x))
        mean = pred.mean.reshape(-1)
        std = pred.stddev.reshape(-1)
        return mean, std





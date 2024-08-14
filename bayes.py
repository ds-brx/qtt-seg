from cost import CostEstimator
from performance import PerfEstimator
from torchvision import transforms
from scipy.stats import norm
import numpy as np
from data import CamVidDataset,train_config
import torch

class Optimiser:
    def __init__(self,configs,max_budget):
        self.configs = configs
        self.max_budget = max_budget
        self.cost_estimator = CostEstimator()
        self.perf_estimator = PerfEstimator()

        self.xi = 0.0
        self.budget_loss = 0
        self.budget_cost = 0

        self.train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,256))
    ])

        self.train_dataset = CamVidDataset(root_dir='/work/dlclarge2/dasb-Camvid/CamVid', split='train', transform=self.train_transform)
        self.val_dataset = CamVidDataset(root_dir='/work/dlclarge2/dasb-Camvid/CamVid', split='val', transform=self.train_transform)
        self.num_classes = 11

    def train(self,b):        
        idx,config,perf_curve,budget,perf_target,meta_feat = self.prepare_data_perf(self.configs,b-1,"train")
        self.perf_estimator.fit_pipeline(config,perf_curve,budget,perf_target,meta_feat)
        
        idx,config,cost_curve,budget,cost_target,meta_feat = self.prepare_data_cost(self.configs,b-1,"train")
        self.cost_estimator.fit_pipeline(config,cost_curve,budget,cost_target,meta_feat)
        
       
    def predict(self,b):
        print("Predicting Performance and Cost...")
        idx,config,perf_curve,budget,_,meta_feat = self.prepare_data_perf(self.configs,b)
        
        mean, std = self.perf_estimator.predict_pipeline(config,perf_curve,budget,meta_feat)

        idx,config,cost_curve,budget,_,meta_feat = self.prepare_data_cost(self.configs,b)
       
        cost = self.cost_estimator.predict_pipeline(config,cost_curve,budget,meta_feat)
        
        return mean,std,cost
    
    def acq(self,budget_loss,mean,std,budget_cost,xi):
        print("Ranking Configs by EI ACQ...")
        mask = std == 0
        std = std + mask * 1.0
        z = (mean - budget_loss - xi) / std
        acq_value = (mean - budget_loss - xi) * norm.cdf(z) + std * norm.pdf(z)
        if isinstance(acq_value, float):
            acq_value = acq_value if mask else 0.0
        else:
            acq_value[mask] = 0.0
        return acq_value / budget_cost

    def suggest_config(self,b):
        self.train(b)
        print("Suggesting Config...")
        mean, std, cost = self.predict(b)
        acq_vals = self.acq(self.budget_loss,mean,std,self.budget_cost,self.xi)
        best_config_id = np.argmax(acq_vals)
        return best_config_id,mean, std, cost

    def prepare(self):
        for config in self.configs:
            print("Preparing Config {}".format(config["idx"]))
            config = self.finetune(config,budget=1)
            self.update_configs(config)

    def finetune(self,config,budget):
        perf,cost = train_config(config,budget,self.num_classes,self.train_dataset,self.val_dataset)
        config["perf_curve"][budget] = perf
        config["cost_curve"][budget] = cost
        self.budget_loss = perf
        self.budget_cost = cost
        return config
    
    def update_configs(self, config):
        idx = config["idx"]
        for conf in self.configs:
            if conf["idx"] == idx:
                conf["perf_curve"] = config["perf_curve"]
                conf["cost_curve"] = config["cost_curve"]
                break

    def prepare_data_perf(self,data,b,flag=None):
        print("Preparing data perf")
        idx = []
        config = []
        budget = []
        perf_curve = []
        perf_target = []
        meta_feat = []
        for i in range(len(data)):
            
            idx.append(data[i]["idx"])
            config.append(list(data[i]["config"].values()))
            budget.append([b])
            meta_feat.append(data[i]["meta_feat"])
            
            if flag == "train":
                perf_curve.append(data[i]["perf_curve"][:b] + [0] * (self.max_budget - len(data[i]["perf_curve"][:b])))
                perf_target.append(data[i]["perf_curve"][b])
            else:
                perf_curve.append(data[i]["perf_curve"])
        return idx,torch.tensor(config),torch.tensor(perf_curve),torch.tensor(budget),torch.tensor(perf_target),torch.tensor(meta_feat)

    def prepare_data_cost(self,data,b,flag=None):
        print("Preparing data cost")
        idx = []
        config = []
        budget = []
        cost_curve = []
        cost_target = []
        meta_feat = []
        for i in range(len(data)):
            idx.append(data[i]["idx"])
            config.append(list(data[i]["config"].values()))
            budget.append([b])
            meta_feat.append(data[i]["meta_feat"])
            if flag == "train":
                cost_curve.append(data[i]["cost_curve"][:b] + [0] * (self.max_budget - len(data[i]["cost_curve"][:b])))
                cost_target.append(data[i]["cost_curve"][b])
            else:
                cost_curve.append(data[i]["cost_curve"])
        
        return idx,torch.tensor(config),torch.tensor(cost_curve),torch.tensor(budget),torch.tensor(cost_target),torch.tensor(meta_feat)

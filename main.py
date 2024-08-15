
import torchvision.models as models

from bayes import Optimiser
from config import sample_configurations

if __name__ == "__main__":

    hp_file = "hp.json"
    num_configs = 1
    max_budget = 10

    configs = sample_configurations(hp_file,num_configs,max_budget)


    opt = Optimiser(configs,max_budget)
    
    for b in range(max_budget):
        if b == 0:
            print("Initialising all configs on 1 budget")
            opt.prepare() #evaluate all configs on 1 epoch 
        else:
            print("Starting Search...")
            print("Budget: ", b)
            config_id,mean, std, cost = opt.suggest_config(b)
            print("Suggest Configeration:\n")
            print("Budget: {} Config ID: {} Loss: {} Cost: {}".format(b,config_id,mean,cost))
            config = opt.finetune(configs[config_id],b)
            opt.update_configs(config)
            

    print("Search Over")
    print("Winner Config: ", config)

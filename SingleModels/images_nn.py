if __name__ == '__main__' and not __package__:
    import sys
    sys.path.insert(0, "/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/")
    __package__ = 'SingleModels'
import torch


import wandb
import numpy as np
import pandas as pd


from .models.image import ImageClassification
from utils.data_loaders import ImageDataset
from .train_model.image_training import img_train, evaluate_img
from utils.global_functions import arg_parse , hidden_layer_count


def runModel(df, param_dict , model_param):
    df_train = df[df['split'] == "train"] # 9989 rows of training data
    df_test = df[df['split'] == "test"] # 2610 rows of testing data
    df_val = df[df['split'] == "val"] # 1106 rows of validation data 

    train, val , test = ImageDataset(df_train), ImageDataset(df_val) , ImageDataset(df_test)
    model = ImageClassification(model_param)
    

    wandb.watch(model, log = "all")

    epoch = param_dict['epoch']
    lr = param_dict['lr']
    batch_size = param_dict['batch_size']
    weight_decay = param_dict['weight_decay']
    clip = param_dict['clip']
    T_max = param_dict['T_max']
    patience = param_dict['patience']


    print(f" Right before entering training loop \n param_dict = {param_dict} \n model_param = {model_param}")
    model = img_train(model, train, val, lr, epoch , batch_size, weight_decay, T_max , patience , clip )
    evaluate_img(model, test )

def main():
    project_name = "LSTM_text"
    args =  arg_parse(project_name)
    # wandb.init(project="Emotion_" + args.dataset + "_" + args.model, entity="ddi" , config = args)
    wandb.init(project=project_name, entity="ddi" , config = args)
    config = wandb.config
    # print(config)
    # Set random seeds
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    
    param_dict = {
        'epoch':config.epoch ,
        'patience':config.patience ,
        'lr': config.learning_rate ,
        'clip': config.clip ,
        'batch_size':config.batch_size ,
        'weight_decay':config.weight_decay ,
        'model': config.model,
        'T_max':config.T_max ,
    }

    model_param = {
        'input_dim':config.input_dim ,
        'output_dim':config.output_dim ,
        'lstm_layers':config.lstm_layers ,
        'hidden_layers':hidden_layer_count(config.hidden_layers) ,
    }

    
    df = pd.read_pickle(f"{args.dataset}.pkl")
    

    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param}")
    runModel(df , param_dict , model_param)
    
if __name__ == '__main__':
    main()




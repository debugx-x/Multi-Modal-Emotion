import sys
sys.path.insert(0,"/home/prsood/projects/def-whkchun/prsood/multi-modal-emotion/") 
__package__ = 'SingleModels'

from .train_model.text_training import train_text_network, evaluate_text
from .models.text import BertClassifier, LSTMClassifier
import wandb

from utils.data_loaders import BertDataset , LstmDataset
from utils.global_functions import arg_parse , CPU_Unpickler
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.global_functions import arg_parse , hidden_layer_count , Metrics
from sklearn.model_selection import train_test_split


def prepare_dataloader(df , batch_size, pin_memory=False, num_workers=4):
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our 
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each 
    """
    dataset = BertDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, 
            num_workers=num_workers, drop_last=False, shuffle=False)
    return dataloader

def runModel( rank , df_train , df_val ,param_dict , model_param , wandb):
    """
    Start by getting all the required values from our dictionary
    Then when all the stuff is done, we start to apply our multi-processing to our model and start it up
    """    
    max_len = 92 # just max number of tokens from LSTM    keep this line in here somewhere
    epoch = param_dict['epoch']
    lr = param_dict['lr']
    patience = param_dict['patience']
    clip = param_dict['clip']
    T_max = param_dict['T_max']
    batch_size = param_dict['batch_size']
    weight_decay = param_dict['weight_decay']
    weights = param_dict['weights']
    label2id = param_dict['label2id']
    id2label = param_dict['id2label']

    num_labels = model_param['output_dim']
    input_dim = model_param['input_dim']
    lstm_layers = model_param['lstm_layers']
    hidden_layers = model_param['hidden_layers']

    criterion = torch.nn.CrossEntropyLoss().to(rank)
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = rank)
    df_train = prepare_dataloader(df_train, batch_size = batch_size )
    df_val = prepare_dataloader(df_val, batch_size = batch_size )

    if param_dict['model'] == "Bert":
        model = BertClassifier(model_param)  
    else:
        #train, val , test = LstmDataset(df_train , max_len=max_len), LstmDataset(df_val , max_len=max_len) , LstmDataset(df_test , max_len=max_len)
        glove_vec = df_train.get_glove_vocab()
        model = LSTMClassifier(glove_vec , model_param)

    wandb.watch(model, log = "all")
    train_text_network(wandb , model, df_train, df_val, criterion , lr, epoch ,  weight_decay,T_max, Metric , patience , clip )

def runTest( rank, df_test , param_dict , model_param , wandb):
    label2id = param_dict['label2id']
    id2label = param_dict['id2label']
    batch_size = param_dict['batch_size']

    num_labels = model_param['output_dim']
    input_dim = model_param['input_dim']
    lstm_layers = model_param['lstm_layers']
    hidden_layers = model_param['hidden_layers']

    criterion = None
    Metric = Metrics(num_classes = num_labels, id2label = id2label , rank = rank)
    df_test = prepare_dataloader(df_test , batch_size = batch_size , check = False )
    
    if param_dict['model'] == "Bert":
        model = BertClassifier(model_param)  
    else:
        #train, val , test = LstmDataset(df_train , max_len=max_len), LstmDataset(df_val , max_len=max_len) , LstmDataset(df_test , max_len=max_len)
        glove_vec = df_test.get_glove_vocab()
        model = LSTMClassifier(glove_vec , model_param)

    print("loading models state dict now" , flush = True)
    model.load_state_dict(torch.load(f'../ModelFolder/{wandb.name}.pt')) 
    wandb.watch(model, log = "all")
    evaluate_text(wandb, model, df_test, Metric)

def main():
    project_name = "MLP_test_text"
    args =  arg_parse(project_name)
    run = wandb.init(project=project_name, entity="ddi" , config = args)
    config = wandb.config
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
        # Need to add the hidden layer count for each modality for their hidden layers 
        'hidden_layers':hidden_layer_count(config.hidden_layers) ,
    }

    df = pd.read_pickle(f"{args.dataset}.pkl")

    """
    Due to data imbalance we are going to reweigh our CrossEntropyLoss
    To do this we calculate 1 - (num_class/len(df)) the rest of the functions are just to order them properly and then convert to a tensor
    """
    weights = torch.Tensor(list(dict(sorted((dict(1 - (df['emotion'].value_counts()/len(df))).items()))).values()))
    # weights = torch.Tensor([1,1,1,1,1,1,1])

    df_train, df_test, _, __ = train_test_split(df, df["emotion"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df["emotion"])
    df_train, df_val, _, __ = train_test_split(df_train, df_train["emotion"], test_size = 0.25, random_state = param_dict['seed'] , stratify=df_train["emotion"])

    label2id = df.drop_duplicates('label').set_index('label').to_dict()['emotion']
    id2label = {v: k for k, v in label2id.items()}
    
    param_dict['weights'] = weights
    param_dict['label2id'] = label2id
    param_dict['id2label'] = id2label

    print(f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {args.dataset}")

    world_size = torch.cuda.device_count()
    print(f"world_size = {world_size}")

    runModel("cuda",df_train , df_val ,param_dict , model_param , run )
    runTest("cuda", df_test ,param_dict , model_param , run )
    
if __name__ == '__main__':
    main()






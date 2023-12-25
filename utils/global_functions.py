from re import X
import torch
from argparse import ArgumentParser
import pickle
import io
import numpy as np
from torchmetrics.classification import MulticlassF1Score , MulticlassRecall , MulticlassPrecision , MulticlassAccuracy , MulticlassConfusionMatrix

def pool(input : np.array , mode : str) -> np.array:
    """
    Supported modes are 'mean', 'max' and 'median'
    Given an array with one dimension, we take the mean max or
    median of it and return it
    """
    if mode == 'mean':
        return torch.Tensor(input.mean(0))
    elif mode == 'max':
        return torch.Tensor(input.max(0))
    elif mode == 'median':
        return torch.Tensor(np.median(input,0))
    else:
        raise NotImplementedError("The supported modes are 'mean', 'max' and 'median'")

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class Metrics:
    """
    Here is where we compute the scores using torch metric 
    """
    def __init__(self, num_classes : int , id2label : dict ,rank , top_k = 1 , average = 'none' , multidim_average='global', ignore_index=None, validate_args=False) -> None:
        self.num_classes = num_classes
        self.top_k = top_k
        self.average = average
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.lol = None
        self.rank = rank
        if self.rank == None:
            self.lol =  "cpu"
        else:
            self.lol = "cuda"

        self.id2label = id2label

        self.confusionMatrix = MulticlassConfusionMatrix( num_classes = self.num_classes , ignore_index=self.ignore_index , normalize='none' , validate_args=self.validate_args).to(device=self.lol)

        self.multiF1 = MulticlassF1Score(self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.multiRec = MulticlassRecall(self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.multiPrec = MulticlassPrecision(self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.multiAcc = MulticlassAccuracy(self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        
        self.scalarF1 = MulticlassF1Score(self.num_classes, self.top_k, 'weighted', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarRec = MulticlassRecall(self.num_classes, self.top_k, 'weighted', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarPrec = MulticlassPrecision(self.num_classes, self.top_k, 'weighted', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)
        self.scalarAcc = MulticlassAccuracy(self.num_classes, self.top_k, 'weighted', self.multidim_average, self.ignore_index, self.validate_args).to(device=self.lol)

    
    def update_metrics(self, preds , target ):
        self.multiF1.update(preds,target)
        self.multiRec.update(preds,target)
        self.multiPrec.update(preds,target)
        self.multiAcc.update(preds,target)
        self.scalarF1.update(preds,target)
        self.scalarRec.update(preds,target)
        self.scalarAcc.update(preds,target)
        self.scalarPrec.update(preds,target)
        self.confusionMatrix.update(preds,target)
        


    def reset_metrics(self):
        self.multiF1.reset()
        self.multiRec.reset()
        self.multiPrec.reset()
        self.multiAcc.reset()
        self.scalarF1.reset()
        self.scalarRec.reset()
        self.scalarAcc.reset()
        self.scalarPrec.reset()
        self.confusionMatrix.reset()

    
    def compute_scores(self, name):
        scalarF1 = self.scalarF1.compute()
        scalarRec = self.scalarRec.compute()
        scalarAccuracy = self.scalarAcc.compute()
        scalarPrec = self.scalarPrec.compute()
        multiF1 = self.multiF1.compute()
        multiRec = self.multiRec.compute()
        multiPrec = self.multiPrec.compute()
        multiAccuracy = self.multiAcc.compute()
        confusionMatrix = self.confusionMatrix.compute()
        return { name +  "/" + "multiAcc/" + self.id2label[v]: k.item() for v, k in enumerate(multiAccuracy)}, { name +  "/" + "multiF1/" + self.id2label[v]: k.item() for v, k in enumerate(multiF1)} , { name +  "/" + "multiRec/" + self.id2label[v]: k.item() for v, k in enumerate(multiRec)} , { name +  "/" + "multiPrec/" + self.id2label[v]: k.item() for v, k in enumerate(multiPrec)}, scalarAccuracy,scalarF1, scalarRec, scalarPrec , confusionMatrix

def hidden_layer_count(string):
    """
    checks that dimensions of hidden layers are consistent
    """
    x = string.split(',')
    if len(x) == 1 or len(x)%2 == 0:
        return list(map(int, x))
    raise ArgumentParser.ArgumentTypeError(f'Missing a dimension in hidden layers, Need to input an even amount of dimensions, that is greater then 1 : {string}')


def arg_parse(description):
    """
    description : str , is the name you want to give to the parser usually the model_modality used
    """
   # pdb.set_trace()
    parser = ArgumentParser(description= f" Run experiments on {description} ")

    # parser.add_argument("--")
    parser.add_argument("--learning_rate" , "-l" , help="Set the learning rate"  , default=0.001, type=float)
    parser.add_argument("--epoch"  , "-e", help="Set the number of epochs"  , default=3, type = int)
    parser.add_argument("--batch_size", "-b", help="Set the batch_size"  , default=16, type=int)
    parser.add_argument("--weight_decay", "-w", help="Set the weight_decay" , default=0.000001, type=float)
    parser.add_argument("--clip", "-c", help="Set the gradient clip" , default=1.0, type=float)
    parser.add_argument("--patience", "-p", help="Set the patience" , default=10.0, type=float)
    parser.add_argument("--T_max", "-t", help="Set the gradient T_max" , default=10, type=int)

    # Set the seed
    parser.add_argument("--seed", "-s", help="Set the random seed" , default=32, type=int)
    
    # These are values in the yaml that we set
    parser.add_argument("--dataset"  , "-d", help="The dataset we are using currently, or the folder the dataset is inside") 
    parser.add_argument("--model"  , "-m", help="The model we are using currently") 

    # These are args that we use as input to the model
    parser.add_argument("--input_dim", "-z", help="Set the input dimension", default=2 ,type=int)
    parser.add_argument("--output_dim", "-y", help="Set the output dimension" , default=7, type=int)
    parser.add_argument("--lstm_layers"  , "-ll", help="set number of LSTM layers" , default=1  ,  type=int) 
    parser.add_argument("--hidden_layers"  , "-o", help="values corresponding to each hidden layer" , default="32,32" , type = str)
    return parser.parse_args()

 


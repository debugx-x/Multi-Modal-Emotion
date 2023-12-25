import pdb
import numpy as np
import torch
import torch.nn as nn
from torch import nn
import sys
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import torchaudio
from utils.global_functions import pool


def speech_file_to_array_fn(path , target_sampling_rate):

    # path = path[6:]
    
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return pool(speech , "mean")

def pad_and_process_values(input_val, processor , target_sampling_rate):
    # m = 166571 # takes 8 minutes to get this value on a pre-processing step with 10K data points
    m = max(map(np.shape , input_val))[0]
    inp = []
    for matrix in input_val:
        n = matrix.shape[0]
        mat = np.pad(matrix, (0, m-n), 'constant')
        inp.append(mat)
    

    result = processor(inp, sampling_rate=target_sampling_rate  , max_length=int(target_sampling_rate * 20.0), truncation=True)

    result = result['input_values']

    return result 


def collate_batch(batch): # batch is a pseudo pandas array of two columns
    """
    Here we are going to take some raw-input and pre-process them into what we need
    So we can see all the steps in the ML as they go along
    """
    # print("in collate batch broski")
    # model_path = "facebook/wav2vec2-large-960h"
    model_path = "facebook/wav2vec2-base-960h"
    # feature_extractor =  Wav2Vec2FeatureExtractor.from_pretrained(model_path) 
    # target_sampling_rate = feature_extractor.sampling_rate

    processor = Wav2Vec2Processor.from_pretrained(model_path)
    target_sampling_rate = processor.feature_extractor.sampling_rate

    speech_list = []
    label_list = []
   
    for (input_path , label) in batch: # change the input path to values that the model can use
        # for audios case we are changing the path to a like a matrix of size 100K
        speech_list.append(speech_file_to_array_fn(input_path , target_sampling_rate))
        label_list.append(label)

    speech_list = pad_and_process_values(speech_list , processor , target_sampling_rate )
    # result = processor(speech_list, sampling_rate=target_sampling_rate , padding = True , max_length=int(target_sampling_rate * 20.0), truncation=True)
    # speech_list = result['input_values']
    

    return torch.Tensor(np.array(speech_list)), torch.Tensor(np.array(label_list))
   


class AudioMLP(nn.Module):
    """
    This MLP functions on the audio files associated with MUStARD.
    """

    def __init__(
        self,
        args, 
        activation: str = "tanh",
        prob_dist: str = "sigmoid",
    ):
        super(AudioMLP, self).__init__()
        """
        Initialize the network.

        :input_dim (int): Set the size of the vocabulary.
        :output_dim (int): Set the output dimensionality (i.e. the number of classes).
        :activation (str, default = 'tanh'): Set the non-linear activation function.
        :prob_dist (str, default = 'sigmoid'): Set the function to compute the pseudo-probability.
        """


        self.hidden_dim = args['hidden_layers']

        
        self.hidden_layers = [] # array of all hidden layers
        self.input_dim = args["input_dim"]
        self.output_dim = args["output_dim"]


        self.in_layer = nn.Embedding(self.input_dim , self.hidden_dim[0]) # 32 , 64 , 42 , 20

        # i = -1: 
        #   (input , 32)  
        # START LOOP:
        # i = 0:
        #   (32 , 64) 
        # i = 1:
        #   (64 , 42) 
        # i = 2:
        #   (42 , 20) 
        # END LOOP:
        # i = 3:
        #   (20 , output_dim)

        for i in range(0,len(self.hidden_dim) - 1): # so we get to the second last value, since last value corresponds to the out layer
            self.hidden_layers.append(
                nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1])
            )

        self.out = nn.Linear(self.hidden_dim[-1], self.output_dim) 

        activation = activation.lower()
        if activation == "tanh":
            self.nonlinearity = nn.Tanh()
        elif activation == "relu":
            self.nonlinearity = nn.ReLU()
        else:
            raise (ValueError("Unknown non-linearity chosen."))

        prob_dist = prob_dist.lower()
        if prob_dist == "sigmoid":
            self.prob_dist = nn.Sigmoid() # tried logSigmoid and regular sigmoid
        elif prob_dist == "softmax":
            self.prob_dist = nn.LogSoftmax()

    def forward(self, input_doc):
        """
        Run the forward step of the algorithm.

        :input_doc: The input document to be classified or trained on.
        """

        # Pass document represenation through the network
        output = self.in_layer(input_doc)  # Input_dim -> 32 dim
        output = self.nonlinearity(output)

        for layer in self.hidden_layers:
            output = layer(output)
            output = self.nonlinearity(output)

        output = self.out(output)  # 64 -> 1

        # Create pseudo-probabilities for outout
        prob_dist = self.prob_dist(output)
        # prob_dist = prob_dist.squeeze(0)
        # prob_dist = torch.tensor(list(map(max, prob_dist))) # since its a n*2 array i tried to just get the max of each row, and make that into a list, and then tensor that

        return prob_dist.squeeze(0)

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.hidden_layers = config.hidden_layers[0]
        self.dropout = nn.Dropout(config.final_dropout)
        self.dense = nn.Linear( config.hidden_size ,  self.hidden_layers)
        self.out_proj = nn.Linear(self.hidden_layers, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config , rank):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config
        self.rank = rank
        self.lol = None
        if self.rank == None:
            self.lol =  "cpu"
        else:
            self.lol = "cuda"



        self.wav2vec2 = Wav2Vec2Model(config).to(device=self.lol)
        self.classifier = Wav2Vec2ClassificationHead(config).to(device=self.lol)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None,):
        # print(f"We are on GPU {self.rank} with input shape {input_values.shape}")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("before wav2vec2" , flush = True)
        outputs = self.wav2vec2(input_values.to(device=self.lol),attention_mask=attention_mask,output_attentions=output_attentions,output_hidden_states=output_hidden_states,return_dict=return_dict)
        # print("after wav2vec2" , flush = True)
        # outputs is made up of 2 elements hidden states at pos 0 and extracted features at pos 1     
        hidden_states = outputs[0]
        # print("after hidden states" , flush = True)
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode) # 3 dim matrix and we want to make it 2 dim
        # print("before classifier" , flush = True)
        logits = self.classifier(hidden_states)
        # we are NOT running a softmax or sigmoid as our last layer, we dont need it 
        return logits

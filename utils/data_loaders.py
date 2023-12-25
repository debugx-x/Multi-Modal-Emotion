import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, vocab

class LstmAudioDataset(Dataset):
    def __init__(self, df , max_len):
        glove = GloVe(name='840B', dim=300 , cache = ".vector_cache")
        self.glove_vocab = vocab(glove.stoi)


        specials = {('[unk]', 0), ('[sep]', 1), ('[pad]', 2)} # 0 , 1 , 2
        # Weird formatting issue so PAD is actually 2
        # Also tokenizer makes everything lowercase, so these have to be as well
        for tok, ix in specials:
            self.glove_vocab.insert_token(tok , ix)

        self.glove_vocab.set_default_index(0)

        tokenizer = get_tokenizer("basic_english")

        try:
            self.labels = df['y'].values.tolist()
        except:
            self.labels = df['label'].values.tolist()

        self.glove_vocab.vectors = glove.get_vecs_by_tokens(self.glove_vocab.get_itos())

    
       
        self.texts = []
        for text in df['text']:
            lst_tokens = tokenizer(text)
            lst_tokens += ['[pad]'] * (max_len - len(lst_tokens)) # found out 128 is the max
            self.texts.append(
                torch.LongTensor(
                    self.glove_vocab.lookup_indices(
                        tokens=lst_tokens
                        )
                )
            )

        self.audio_features = df['path'].values


    def get_glove_vocab(self): return self.glove_vocab

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx): return {"Lstm":self.texts[idx] , "Audio":self.audio_features[idx]}, np.array(self.labels[idx])

class BertAudioDataset(Dataset):
    """
    Taking in both the text of the pandas file and the path of audio file
    getting their path 
    """
    def __init__(self, df , max_len):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        try:
            self.labels = df['y'].values.tolist()
        except:
            self.labels = df['label'].values.tolist()
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = max_len, truncation=True,
                                return_tensors="pt") for text in df['text']]
        
        self.audio_features = df['path'].values


    def __len__(self): return len(self.labels)

    def __getitem__(self, idx): return {"Bert":self.texts[idx] , "Audio":self.audio_features[idx]}, np.array(self.labels[idx])

class VisualDataset(Dataset):
    """A basic dataset where the underlying data is a list of (x,y) tuples. Data
    returned from the dataset should be a (transform(x), y) tuple.
    Args:
    source      -- a list of (x,y) data samples
    transform   -- a torchvision.transforms transform
    """
    def __init__(self, df):
        super(VisualDataset, self).__init__()
        self.video_path = df['video_path'].values
        self.label = df['label'].values

    def __len__(self): return len(self.label)
    
    def __getitem__(self, idx): return self.video_path[idx] , np.array(self.label[idx])

class ImageDataset(Dataset):
    """A basic dataset where the underlying data is a list of (x,y) tuples. Data
    returned from the dataset should be a (transform(x), y) tuple.
    Args:
    source      -- a list of (x,y) data samples
    transform   -- a torchvision.transforms transform
    """
    def __init__(self, df):
        super(ImageDataset, self).__init__()
        self.img_path = df['img_path'].values
        self.label = df['label'].values

    def __len__(self): return len(self.label)
    
    def __getitem__(self, idx): return self.img_path[idx] , np.array(self.label[idx])

class Wav2VecAudioDataset(Dataset):
    def __init__(self, df):
        """
        Initialize the dataset loader.

        :data: The dataset to be loaded.
        :labels: The labels for the dataset."""

        self.labels = df['emotion'].values 
        # Want a tensor of all the features d
        self.audio_features = df['path'].values

        self.df = df
        #copy unique df and the id2label functinos to whichever dataloader ytou are using
        self.unique_df = self.df.drop_duplicates('label')


        assert self.audio_features.shape == self.labels.shape  # TODO Double check that this asserts the right dims.
        self.length = self.audio_features.shape[0]

    
    def label2id(self):
        return self.unique_df.set_index('label').to_dict()['emotion']

    def id2label(self):
        return self.unique_df.set_index('emotion').to_dict()['label']

    def __getitem__(self, idx: int): return self.audio_features[idx], np.array(self.labels[idx])

    def __len__(self): return self.length
    

class LstmDataset(Dataset):
    """
    Load text dataset for lstm processing.
    """

    def __init__(self, df , max_len):
        glove = GloVe(name='840B', dim=300 , cache = ".vector_cache")
        self.glove_vocab = vocab(glove.stoi)


        specials = {('[unk]', 0), ('[sep]', 1), ('[pad]', 2)} # 0 , 1 , 2
        # Weird formatting issue so PAD is actually 2
        # Also tokenizer makes everything lowercase, so these have to be as well
        for tok, ix in specials:
            self.glove_vocab.insert_token(tok , ix)

        self.glove_vocab.set_default_index(0)

        tokenizer = get_tokenizer("basic_english")

        try:
            self.labels = df['y'].values.tolist()
        except:
            self.labels = df['label'].values.tolist()

        self.glove_vocab.vectors = glove.get_vecs_by_tokens(self.glove_vocab.get_itos())

    
       
        self.texts = []
        for text in df['text']:
            lst_tokens = tokenizer(text)
            lst_tokens += ['[pad]'] * (max_len - len(lst_tokens)) # found out 128 is the max
            self.texts.append(
                torch.LongTensor(
                    self.glove_vocab.lookup_indices(
                        tokens=lst_tokens
                        )
                )
            )

    def get_glove_vocab(self): return self.glove_vocab

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx): return self.texts[idx], np.array(self.labels[idx])
class BertDataset(Dataset):
    """
    Load text dataset for BERT processing.
    """

    def __init__(self, df , max_len):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        

        try:
            self.labels = df['y'].values.tolist()
        except:
            self.labels = df['label'].values.tolist()
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = max_len, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self): return len(self.labels)


    def __getitem__(self, idx): return self.texts[idx], np.array(self.labels[idx])

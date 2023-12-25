import torch
from torch import nn
from tqdm import tqdm
import wandb
from utils.global_functions import compute_scores
from utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR
import pdb
import warnings
from ..models.image import collate_batch
import numpy as np


def get_statistics(input,label,model,criterion,total_loss,total_acc,truth,preds,device = None):
    batch_loss = None
    label = label.float()

    
    output = model(input).float() # input is torch.Size([16, 3, 32, 32])
    # output is torch.Size([32])

    # Batch size is 16

    if criterion is not None:
        batch_loss = criterion(output, label)
        total_loss += batch_loss.item()
    
    truth.extend(label.reshape(-1).tolist())
    preds.extend(output.round().tolist())

    acc = (output.round() == label.reshape(-1)).sum().item()
    total_acc += acc

    return batch_loss , total_loss , total_acc , truth , preds



def one_epoch(train_dataloader , model , criterion , optimizer, clip , device = None):
    total_acc_train = 0
    total_loss_train = 0
    preds = []
    truth = []
    
    for train_input, train_label in tqdm(train_dataloader , desc="training"):
        train_batch_loss , total_loss_train , total_acc_train , truth , preds = get_statistics(train_input , train_label , model , criterion , total_loss_train , total_acc_train , truth , preds , device)
        train_batch_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    f1 , prec , recall = compute_scores(truth , preds)
    

    return model , optimizer , train_batch_loss,total_loss_train/len(truth),total_acc_train/len(truth),prec,recall,f1


def validate(val_dataloader , model , criterion, device = None):
    preds = []
    truth = []
    total_acc_val = 0
    total_loss_val = 0
    with torch.no_grad():

        for val_input, val_label in tqdm(val_dataloader, desc="validation"):

            val_batch_loss , total_loss_val , total_acc_val , truth , preds = get_statistics(val_input , val_label , model , criterion , total_loss_val , total_acc_val , truth , preds , device)
                     
    f1 , prec , recall = compute_scores(truth , preds)

    return val_batch_loss,total_loss_val/len(truth),total_acc_val/len(truth),prec,recall,f1 

    
   


def img_train(model, train_data, val_data,learning_rate, epochs , batch_size , weight_decay , T_max , patience , clip):
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True  , collate_fn = collate_batch)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True  , collate_fn = collate_batch)

    # pdb.set_trace()

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # if model.__class__.__name__ == "BertClassifier":
    #     from transformers.optimization import AdamW
    # else:
    from torch.optim import AdamW

    criterion = nn.BCELoss(reduction="mean") # This is different 
    optimizer = AdamW(model.parameters(), lr= learning_rate, weight_decay=weight_decay)

    # if use_cuda:
    #     model = model.cuda()
    #     criterion = criterion.cuda()

    
    # earlystop = EarlyStopping("",model,patience,model_name=model.__class__.__name__)

    scheduler = CosineAnnealingLR(optimizer , T_max=T_max)  # To prevent fitting to local minima != global minima

    for epoch_num in tqdm(range(epochs), desc="epochs"):
        model.train()


        wandb.log({"epoch":  epoch_num})

        optimizer.zero_grad()  # Zero out gradients before each epoch.
        
        #model,optimizer,train_batch_loss,train_loss,train_acc,prec,recall,f1 = one_epoch(train_dataloader , device , model , criterion , optimizer , clip) 
        _, _,train_batch_loss,train_loss,train_acc,prec,recall,f1 = one_epoch(train_dataloader, model , criterion , optimizer , clip , device = None)

        scheduler.step()
        wandb.log(
            {
                "train/batch_loss": train_batch_loss,

                "train/train_loss": train_loss,
                "train/acc":train_acc,
                
                "train/precision": prec,
                "train/recall" : recall,
                "train/f1-score": f1,
                }
            )           
        model.eval()
        val_batch_loss,val_loss,val_acc,prec,recall,f1 =  validate(val_dataloader, model , criterion , device = None )

        wandb.log(
            {
                "val/batch_loss": val_batch_loss,

                "val/total_loss_val": val_loss,
                "val/total_acc_val": val_acc,
                
                "val/precision": prec,
                "val/recall": recall,
                "val/f1-score": f1,
                }
            )

    return model

def evaluate_img(model, test_data):
    model.eval()

    test_dataloader = torch.utils.data.DataLoader(test_data , collate_fn = collate_batch)

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")

    # if use_cuda:
    #     model = model.cuda()

    total_acc_test = 0
    preds = []
    truth = []
    with torch.no_grad():

        for test_input, test_label in tqdm(test_dataloader, desc="test"):

            _ , __ , total_acc_test , truth , preds = get_statistics(test_input , test_label , model , None , None , total_acc_test , truth , preds , device = None)
            

    f1 , prec , recall = compute_scores(truth , preds)
    
    wandb.log(
        {
            "test/total_acc_test": total_acc_test / len(truth),
            "test/precision": prec,
            "test/recall": recall,
            "test/f1-score": f1,
            }
        )



